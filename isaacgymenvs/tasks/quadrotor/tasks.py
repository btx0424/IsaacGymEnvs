import torch
import math
import numpy as np
from isaacgym import gymtorch
from gym import spaces
from typing import Any, Dict, Tuple
from .base import QuadrotorBase

def normalize(x: torch.Tensor) -> torch.Tensor:
    return x / (1e-7 + x.norm(dim=-1, keepdim=True))

class OccupationIndependent(QuadrotorBase):
    
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture: bool = False, force_render: bool = False):
        cfg["env"]["numRewards"] = 4 # [distance, collision, progress, capture]
        
        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)
        # task specification
        self.capture_radius: float = cfg.get("captureRadius", 0.3)
        self.success_threshold: int = cfg.get("successThreshold", 50)
        self.target_speed: float = cfg.get("targetSpeed", 1.)
        self.agents = ["predator"]

    def allocate_buffers(self):
        super().allocate_buffers()

        self.captured_steps_buf = torch.zeros(
            (self.num_envs, self.num_agents), device=self.device)
        self.target_distance_buf = torch.zeros(
            (self.num_envs, self.num_agents), device=self.device)

    def reset_buffers(self, env_ids):
        super().reset_buffers(env_ids)
        self.target_distance_buf[env_ids] = 0
        self.captured_steps_buf[env_ids] = 0

    def create_obs_space_and_processor(self, obs_type=None) -> None:
        num_obs = 13*self.num_agents + 13 + 13
        ones = np.ones(num_obs)
        self.obs_space = spaces.Box(-ones*np.inf, ones*np.inf)
        def obs_processor(obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
            obs_tensor = []
            identity = torch.eye(self.num_agents, device=self.device, dtype=bool)
            states_self = self.root_states[:, self.env_actor_index["drone"]]

            states_target = self.root_states[:, self.env_actor_index["target"]]
            states_target[..., :3] = states_target[..., :3] - states_self[..., :3]

            states_all = states_self.unsqueeze(1).repeat(1, self.num_agents, 1, 1)
            states_all[..., :3] = states_all[..., :3] - states_self[..., :3].unsqueeze(2) # (env, agent, agent, 3) - (env, agent, 1, 3)
            states_all = states_all.reshape(self.num_envs, self.num_agents, -1)

            obs_tensor.append(states_target)
            obs_tensor.append(states_all)
            obs_tensor.append(states_self)
            obs_dict["state"] = obs_dict["obs"] = torch.cat(obs_tensor, dim=-1)
            return obs_dict
        self.obs_split = [(1, 13), (self.num_agents, 13), (1, 13)]
        self.obs_processor = obs_processor
        self.state_space = self.obs_space

    def compute_reward_and_reset(self):
        pos, quat, vel, angvel = self.quadrotor_states
            
        contact = self.contact_forces[:, self.env_body_index["base"]]

        target_distance = torch.norm(self.target_pos-pos, dim=-1)
        distance_reward = 1.0 / (1.0 + target_distance ** 2)
        spinnage = torch.abs(angvel[..., 2])
        spinnage_reward = 1.0 / (1.0 + spinnage * spinnage)
        distance_reward = distance_reward + distance_reward * spinnage_reward
        collision_penalty = -contact.any(-1).float()
        progress_reward = self.target_distance_buf - target_distance

        captured: torch.Tensor = target_distance < self.capture_radius
        self.captured_steps_buf[~captured] = 0
        self.captured_steps_buf[captured] += 1

        self.rew_buf[..., 0] = distance_reward
        self.rew_buf[..., 1] = collision_penalty
        self.rew_buf[..., 2] = progress_reward
        self.rew_buf[..., 3] = captured.float()
        
        self.cum_rew_buf.add_(self.rew_buf)
        self.target_distance_buf[:] = target_distance
        
        self.reset_buf.zero_()
        self.reset_buf[target_distance > 3] = 1
        self.reset_buf[pos[..., 2] < 0.1] = 1
        self.reset_buf[self.progress_buf >= self.max_episode_length - 1] = 1

        self.extras["episode"].update({
            "reward/distance": self.cum_rew_buf[..., 0],
            "reward/collision": self.cum_rew_buf[..., 1],
            "reward/progress": self.cum_rew_buf[..., 2],
            "reward/capture": self.cum_rew_buf[..., 3],
            
            "success": (self.captured_steps_buf > self.success_threshold).float(),
        })

    def update_targets(self):
        xy = self.target_pos[..., :2] # (env, target, [x, y])
        angvel = math.pi * 2 / self.max_episode_length * self.target_speed
        rad = torch.atan2(xy[..., 1], xy[..., 0]) + angvel
        next_xy = torch.stack([torch.cos(rad), torch.sin(rad)], dim=-1)
        self.root_linvels[:, self.env_actor_index["target"], :2] = (next_xy - xy) / (self.dt * self.control_freq_inv)

        # apply
        actor_reset_ids = self.sim_actor_index["target"].flatten()
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, self.root_tensor, gymtorch.unwrap_tensor(actor_reset_ids), len(actor_reset_ids))

        # visualize if has viewer
        if self.viewer:
            quadrotor_pos = self.quadrotor_pos[0]
            target_pos = self.target_pos[0].expand_as(quadrotor_pos)
            points = torch.cat([quadrotor_pos, target_pos], dim=-1).cpu().numpy()
            self.viewer_lines.append((points, [0, 1, 0]))

    def pre_physics_step(self, actions: torch.torch.Tensor):
        super().pre_physics_step(actions)
        self.update_targets()

    def reset_actors(self, env_ids):
        super().reset_actors(env_ids)
        spacing = torch.linspace(0, self.max_episode_length, self.num_agents+1, device=self.device)[:-1]
        rad = (self.progress_buf[env_ids].unsqueeze(-1)+spacing)/self.max_episode_length*math.pi*2
        env_positions = self.root_positions[env_ids]
        env_velocities = self.root_linvels[env_ids]
        env_positions[:, self.env_actor_index["target"], 0] = torch.sin(rad)
        env_positions[:, self.env_actor_index["target"], 1] = torch.cos(rad)
        env_positions[:, self.env_actor_index["target"], 2] = 0.5
        env_velocities[:, self.env_actor_index["target"]] = 0
        self.root_positions[env_ids] = env_positions
        self.root_linvels[env_ids] = env_velocities

    def agents_step(self, action_dict: Dict[str, torch.Tensor]) -> Tuple[Dict[str, Tuple[Any, torch.Tensor, torch.Tensor]], Dict[str, Any]]:
        obs_dict, reward, done, info = self.step(action_dict["predator"])
        return {"predator": (obs_dict, reward, done)}, info

    def reset(self) -> Dict[str, torch.Tensor]:
        obs_dict =  super().reset()
        return {"predator": obs_dict}

class PredatorPrey(QuadrotorBase):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture: bool = False, force_render: bool = False):
        raise NotImplementedError
        cfg["env"]["numRewards"] = 3 # [distance, collision, capture]
        
        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)
        # task specification
        self.capture_radius: float = cfg.get("captureRadius", 0.3)
        self.success_threshold: int = cfg.get("successThreshold", 50)
        self.agents = ["predator", "prey"]

    def allocate_buffers(self):
        super().allocate_buffers()

        self.captured_steps_buf = torch.zeros(self.num_envs, device=self.device)
        self.target_distance_buf = torch.zeros(
            (self.num_envs, self.num_agents), device=self.device)

    def reset_buffers(self, env_ids):
        super().reset_buffers(env_ids)
        self.target_distance_buf[env_ids] = 0
        self.captured_steps_buf[env_ids] = 0
        
    def create_obs_space_and_processor(self, obs_type=None) -> None:
        num_obs = 13*self.num_agents + 13 + 13
        ones = np.ones(num_obs)
        self.obs_space = spaces.Box(-ones*np.inf, ones*np.inf)
        def obs_processor(obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
            obs_tensor = []
            identity = torch.eye(self.num_agents, device=self.device, dtype=bool)
            states_self = self.root_states[:, self.env_actor_index["drone"]]

            states_target = self.root_states[:, self.env_actor_index["target"][0]].unsqueeze(1).repeat(1, self.num_agents, 1)
            states_target[..., :3] = states_target[..., :3] - states_self[..., :3]
            assert states_target.shape == states_self.shape

            states_all = states_self.unsqueeze(1).repeat(1, self.num_agents, 1, 1)
            states_all[..., :3] = states_all[..., :3] - states_self[..., :3].unsqueeze(2) # (env, agent, agent, 3) - (env, agent, 1, 3)
            states_all = states_all.reshape(self.num_envs, self.num_agents, -1)

            obs_tensor.append(states_target)
            obs_tensor.append(states_all)
            obs_tensor.append(states_self)
            obs_dict["state"] = obs_dict["obs"] = torch.cat(obs_tensor, dim=-1)
            return obs_dict
        self.obs_split = [(1, 13), (self.num_agents, 13), (1, 13)]
        self.obs_processor = obs_processor
        self.state_space = self.obs_space
    
    def compute_reward_and_reset(self):
        pos, quat, vel, angvel = self.quadrotor_states
            
        contact = self.contact_forces[:, self.env_body_index["base"]]

        target_distance = torch.norm(self.target_pos-pos, dim=-1) # (num_envs, num_agents)
        target_distance_min = target_distance.min(-1)
        distance_reward = torch.mean(1.0 / (1.0 + target_distance_min.values ** 2))
        # spinnage = torch.abs(angvel[..., 2])
        # spinnage_reward = 1.0 / (1.0 + spinnage * spinnage)
        # distance_reward = distance_reward + distance_reward * spinnage_reward
        collision_penalty = contact.any(-1).float()

        captured: torch.Tensor = (target_distance_min.values < self.capture_radius).any(-1)
        self.captured_steps_buf[~captured] = 0
        self.captured_steps_buf[captured] += 1

        self.rew_buf[..., 0] = distance_reward
        self.rew_buf[..., 1] = -collision_penalty
        self.rew_buf[..., 2] = captured.float()
        
        self.cum_rew_buf.add_(self.rew_buf)
        
        self.reset_buf.zero_()
        self.reset_buf[(target_distance > 3).all(-1)] = 1
        self.reset_buf[pos[..., 2] < 0.1] = 1
        self.reset_buf[self.progress_buf >= self.max_episode_length - 1] = 1

        self.extras["episode"].update({
            "reward/distance": self.cum_rew_buf[..., 0],
            "reward/collision": self.cum_rew_buf[..., 1],
            "reward/capture": self.cum_rew_buf[..., 2],
            
            "success": (self.captured_steps_buf > self.success_threshold).float(),
        })
    
    @property
    def target_pos(self) -> torch.Tensor:
        return self.root_positions[:, self.env_actor_index["target"][0]].unsqueeze(1)

    @property
    def target_vel(self) -> torch.Tensor:
        return self.root_linvels[:, self.env_actor_index["target"][0]]

    @target_vel.setter
    def target_vel(self, vel: torch.Tensor):
        self.root_linvels[:, self.env_actor_index["target"][0]] = vel

    def update_targets(self):
        target_pos = self.target_pos
        boundary_pos = target_pos.clone()
        boundary_pos[..., :2] = normalize(boundary_pos[..., :2]) * 2
        force_sources = torch.cat([self.quadrotor_pos, boundary_pos], dim=1)
        distance = torch.norm(target_pos - force_sources, dim=-1, keepdim=True)
        forces = (target_pos - force_sources) / (1e-7 + distance**2)
        target_vel = normalize(torch.mean(forces, dim=-2)) * 0.5
        target_vel[..., 2] *= 0.5
        self.target_vel = target_vel

        # apply
        actor_reset_ids = self.sim_actor_index["target"].flatten()
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, self.root_tensor, gymtorch.unwrap_tensor(actor_reset_ids), len(actor_reset_ids))
    
    def pre_physics_step(self, actions: torch.torch.Tensor):
        super().pre_physics_step(actions)
        self.update_targets()
    
    def reset_actors(self, env_ids):
        super().reset_actors(env_ids)
        spacing = torch.linspace(0, self.max_episode_length, self.num_agents+1, device=self.device)[:-1]
        rad = (self.progress_buf[env_ids].unsqueeze(-1)+spacing)/self.max_episode_length*math.pi*2
        env_positions = self.root_positions[env_ids]
        env_velocities = self.root_linvels[env_ids]
        env_positions[:, self.env_actor_index["target"], 0] = torch.sin(rad)
        env_positions[:, self.env_actor_index["target"], 1] = torch.cos(rad)
        env_positions[:, self.env_actor_index["target"], 2] = 0.5
        env_velocities[:, self.env_actor_index["target"]] = 0
        self.root_positions[env_ids] = env_positions
        self.root_linvels[env_ids] = env_velocities