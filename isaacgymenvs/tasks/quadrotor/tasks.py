import torch
import math
import numpy as np
import logging
from isaacgym import gymapi, gymtorch
from gym import spaces
from typing import Any, Callable, Dict, Sequence, Tuple

from torchrl.data.tensordict.tensordict import TensorDictBase
from torchrl.data import TensorDict, CompositeSpec, NdBoundedTensorSpec, NdUnboundedContinuousTensorSpec

from .base import QuadrotorBase

@torch.jit.script
def normalize(x: torch.Tensor) -> torch.Tensor:
    return x / (1e-7 + torch.norm(x, dim=-1, keepdim=True))

def uniform(size, low: float, high: float, device: torch.device) -> torch.Tensor:
    return torch.rand(size, device=device)*(high-low) + low

def mix(x: torch.Tensor, y: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    return x * a + y * (1 - a)

class TargetEasy(QuadrotorBase):
    agents = ["predator"]

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
        # distance_reward = 1.0 / (1.0 + (target_distance-self.capture_radius).clamp(min=0) ** 2)
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
        self.reset_buf[pos[..., 2] > self.MAX_XYZ[2]] = 1
        self.reset_buf[self.progress_buf >= self.max_episode_length - 1] = 1

        cum_rew = self.cum_rew_buf.clone()
        self.extras["episode"].update({
            "reward/distance": cum_rew[..., 0],
            "reward/collision": cum_rew[..., 1],
            "reward/progress": cum_rew[..., 2],
            "reward/capture": cum_rew[..., 3],
            
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
        obs_dict, reward, done, info = super().agents_step(action_dict["predator"])
        return {"predator": (obs_dict, reward, done)}, info

    def step(self, actions):
        raise
        obs_dict, reward, done, info = super().agents_step(actions)
        return obs_dict["obs"], reward, done, info

    def reset(self) -> Dict[str, TensorDict]:
        obs_dict = super().reset()
        return {"predator": obs_dict}

class TargetHard(QuadrotorBase):
    agent_types = ["predator"]

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture: bool = False, force_render: bool = False):
        cfg["env"]["numRewards"] = 3 # [distance, collision, capture]

        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)
        # task specification
        self.capture_radius: float = cfg.get("captureRadius", 0.3)
        self.success_threshold: int = cfg.get("successThreshold", 50)
        self.target_speed = cfg.get("targetSpeed")
        self.target_speeds = torch.ones(self.num_envs, device=self.device) # all targets have the same speed
        self.boundary_radius  = cfg.get("boundaryRadius", 2.5)
        assert self.boundary_radius > 1
        self.task_spec = cfg.get("taskSpec", ["target_speed"])
        self.shared_reward_mix = cfg.get("rewardMix", 1.) # 1: shared, 0: independent

        def reset_actors(_, env_ids: torch.Tensor):
            self.root_positions[env_ids, self.env_actor_index["target"]] = torch.tensor([0, 0, 0.5], device=self.device)
            self.root_linvels[env_ids, self.env_actor_index["target"]] = 0

            randperm = torch.randperm(len(self.grid_avail), device=self.device)
            num_samples = env_ids.numel() * self.num_agents
            sample_idx = randperm[torch.arange(num_samples).reshape(len(env_ids), self.num_agents)%len(self.grid_avail)]
            index_slice = slice(self.env_actor_index["drone"][0], self.env_actor_index["drone"][-1]+1)
            self.root_positions[env_ids, index_slice] = self.grid_centers[sample_idx]
            if "spawn_pos" in self.task_spec:
                self.task_config["spawn_pos_idx"][env_ids] = sample_idx

        self.on_reset(reset_actors)

        self.task_config = self.extras["task_config"] = TensorDict({
            "spawn_pos_idx": torch.zeros(self.num_envs, self.num_agents, dtype=int, device=self.device),
            "target_speed": self.target_speeds
        }, batch_size=self.num_envs)

        if isinstance(self.target_speed, Sequence):
            assert len(self.target_speed) == 2 and self.target_speed[0] <= self.target_speed[1], "Invalid target speed range"
            logging.info(f"Sample target speed from range: {self.target_speed}")
            def sample_speed(_, env_ids):
                self.target_speeds[env_ids] = uniform(len(env_ids), self.target_speed[0], self.target_speed[1], self.device)
            self.reset_callbacks.append(sample_speed)
        else:
            self.target_speeds *= self.target_speed
        
        num_obs = 6*self.num_boxes + 13*self.num_agents + 13 + 13
        ones = np.ones(num_obs)
        self.obs_space = spaces.Box(-ones*np.inf, ones*np.inf)
        self.obs_split = [(self.num_boxes, 6), (1, 13), (self.num_agents, 13), (1, 13)]
        self.observation_spec = CompositeSpec(**{
            "obs@predator": NdUnboundedContinuousTensorSpec((num_obs,), device=self.device),
            "state": NdUnboundedContinuousTensorSpec((num_obs,), device=self.device)
        }) 
        self.action_spec = NdBoundedTensorSpec(-1, 1, (3,), device=self.device)

    def allocate_buffers(self):
        super().allocate_buffers()
        self.captured_steps_buf = torch.zeros(self.num_envs, device=self.device)
        self.target_distance_buf = torch.zeros(
            (self.num_envs, self.num_agents), device=self.device)
        
        def reset_buffers(_, env_ids):
            self.captured_steps_buf[env_ids] = 0
            self.target_distance_buf[env_ids] = 0
        self.on_reset(reset_buffers)
    
    def compute_reward_and_done(self, tensordict: TensorDictBase):
        pos, quat, vel, angvel = self.quadrotor_states.values()
            
        contact = self.contact_forces[:, self.env_body_index["base"]]

        target_distance = torch.norm(self.target_pos-pos, dim=-1) # (num_envs, num_agents)
        distance_reward = (1.0 / (1.0 + target_distance ** 2)) # (num_envs, num_agents)
        collision_penalty = contact.any(-1).float()

        capture: torch.Tensor = target_distance < self.capture_radius
        captured = capture.any(dim=-1)
        self.captured_steps_buf[~captured] = 0
        self.captured_steps_buf[captured] += 1

        capture_reward = capture.float()
        self.rew_buf[..., 0] = mix(distance_reward.mean(-1, keepdim=True), distance_reward, self.shared_reward_mix)
        self.rew_buf[..., 1] = -collision_penalty
        self.rew_buf[..., 2] = mix(capture_reward.mean(-1, keepdim=True), capture_reward, self.shared_reward_mix)
        
        self.cum_rew_buf.add_(self.rew_buf)
        
        self.reset_buf.zero_()
        # self.reset_buf[(target_distance > 3).all(-1)] = 1
        self.reset_buf[pos[..., 2] < 0.1] = 1
        self.reset_buf[pos[..., 2] > self.MAX_XYZ[2]] = 1
        self.reset_buf[self.progress_buf >= self.max_episode_length - 1] = 1

        cum_reward = self.cum_rew_buf.clone()
        self.extras["episode"].update({
            "reward_distance@predator": cum_reward[..., 0],
            "reward_collision@predator": cum_reward[..., 1],
            "reward_capture@predator": cum_reward[..., 2],
            "success@predator": (self.captured_steps_buf > self.success_threshold).float(),
            "length": self.progress_buf + 1,
        })

        return TensorDict({
            "reward@predator": self.rew_buf.clone(), 
            "done@predator": self.reset_buf.clone().unsqueeze(-1),
            "env_done": self.reset_buf.all(-1)
        }, batch_size=self.batch_size)
    
    def compute_state_and_obs(self, tensordict: TensorDictBase):
        obs_tensor = []
        identity = torch.eye(self.num_agents, device=self.device, dtype=bool)
        states_self = self.root_states[:, self.env_actor_index["drone"]]

        states_target = self.root_states[:, self.env_actor_index["target"]].repeat(1, self.num_agents, 1)
        states_target[..., :3] = states_target[..., :3] - states_self[..., :3]
        assert states_target.shape == states_self.shape

        states_all = states_self.unsqueeze(1).repeat(1, self.num_agents, 1, 1)
        states_all[..., :3] = states_all[..., :3] - states_self[..., :3].unsqueeze(2) # (env, agent, agent, 3) - (env, agent, 1, 3)
        states_all = states_all.reshape(self.num_envs, self.num_agents, -1)

        states_box = self.box_states.repeat(self.num_envs, self.num_agents, 1, 1)
        states_box[..., :3] = states_box[..., :3] - states_self[..., :3].unsqueeze(2)
        states_box = states_box.reshape(self.num_envs, self.num_agents, -1)
        
        obs_tensor.append(states_target)
        obs_tensor.append(states_box)
        obs_tensor.append(states_all)
        obs_tensor.append(states_self)
        obs_tensor = torch.cat(obs_tensor, dim=-1)

        return TensorDict({
            "next_obs@predator": obs_tensor,
            "next_state": obs_tensor,
        }, batch_size=self.num_envs)
              
    def step(self, tensordict: TensorDictBase) -> TensorDictBase:
        step_result = super().step(TensorDict({"actions": tensordict["actions@predator"]}, batch_size=self.num_envs))
        step_result.del_("actions")
        tensordict.update(step_result)
        return tensordict
    
    @property
    def target_pos(self) -> torch.Tensor:
        return self.root_positions[:, self.env_actor_index["target"]]

    @target_pos.setter
    def target_pos(self, pos: torch.Tensor):
        self.root_positions[:, self.env_actor_index["target"]] = pos

    @property
    def target_vel(self) -> torch.Tensor:
        return self.root_linvels[:, self.env_actor_index["target"]]

    @target_vel.setter
    def target_vel(self, vel: torch.Tensor):
        self.root_linvels[:, self.env_actor_index["target"]] = vel

    def pre_physics_step(self, actions: torch.torch.Tensor):
        super().pre_physics_step(actions)
        
        # update targets
        target_pos = self.target_pos # (num_envs, num_targets, 3)
        # imaginary predator at the boundary
        boundary_pos = target_pos.clone().view(self.num_envs, self.num_targets, 1, 3)
        boundary_pos[..., :2] = normalize(boundary_pos[..., :2]) * self.boundary_radius
        quadrotor_pos = self.quadrotor_pos \
            .view(self.num_envs, 1, self.num_agents, 3) \
            .expand(self.num_envs, self.num_targets, self.num_agents, 3)
        force_sources = torch.cat([quadrotor_pos, boundary_pos], dim=-2) # (num_envs, num_targets, num_agents+1, 3)
        d = target_pos.view(self.num_envs, self.num_targets, 1, 3) - force_sources
        distance = torch.norm(d, dim=-1, keepdim=True)
        forces = d / (1e-7 + distance**2)
        target_vel = normalize(torch.mean(forces, dim=-2)) * self.target_speeds.view(-1, 1, 1)
        
        target_pos[..., 2].clamp_(0.2, self.MAX_XYZ[2]-0.2)
        self.target_pos = target_pos # necessary?
        self.target_vel = target_vel

        # apply
        actor_reset_ids = self.sim_actor_index["target"].flatten()
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, self.root_tensor, gymtorch.unwrap_tensor(actor_reset_ids), len(actor_reset_ids))
        
        # visualize if has viewer
        if self.viewer:
            quadrotor_pos = self.quadrotor_pos[0]
            target_pos = self.target_pos[0].expand_as(quadrotor_pos)
            points = torch.cat([quadrotor_pos, target_pos], dim=-1).cpu().numpy()
            self.viewer_lines.append((points, [[0, 1, 0]]*len(points)))
    
    def set_tasks(self, 
            env_ids: torch.Tensor, 
            task_config: TensorDictBase,
            env_states: torch.Tensor):
        if "spawn_pos" in self.task_spec:
            spawn_pos_idx = task_config["spawn_pos_idx"]
            drone_positions = env_states[:, self.env_actor_index["drone"], :3]
            drone_positions[:] = self.grid_centers[spawn_pos_idx]
        if "target_speed" in self.task_spec:
            target_speed = task_config["target_speed"]
            self.target_speeds[env_ids] = target_speed.squeeze()

    def get_dummy_policy(self, *args, **kwargs) -> Callable:
        def dummy_policy(tensordict: TensorDict):
            target_pos = tensordict["obs@predator"][..., :3]
            tensordict["actions@predator"] = target_pos
            tensordict["value@predator"] = torch.zeros(tensordict["obs@predator"].shape[:-1] + (1,), device=self.device)
            tensordict["action_log_prob@predator"] = torch.zeros(tensordict["obs@predator"].shape[:-1] + (1,), device=self.device)
            return tensordict
        return dummy_policy
    
class PredatorPrey(QuadrotorBase):

    agent_types = ["predator", "prey"]

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture: bool = False, force_render: bool = False):
        cfg["env"]["numRewards"] = 3 # [distance, collision, capture]
        cfg["env"]["numTargets"] = 0

        self.num_predators = cfg["env"].get("numPredators", 2)
        self.num_preys = cfg["env"].get("numPreys", 1)
        cfg["env"]["numAgents"] =  self.num_predators + self.num_preys
        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)
        # task specification
        self.capture_radius: float = cfg.get("captureRadius", 0.3)
        self.success_threshold: int = cfg.get("successThreshold", 50)

        self.predator_index = slice(None, -1)
        self.prey_index = slice(-1, None)
        self.env_predator_index = self.env_actor_index["drone"][self.predator_index]
        self.env_prey_index = self.env_actor_index["drone"][self.prey_index]
        self.env_drone_slice = slice(self.env_actor_index["drone"][0], self.env_actor_index["drone"][-1] + 1)
        self.obs_split = [(self.num_boxes, 6), (self.num_preys, 13), (self.num_predators, 13), (1, 13)]
        obs_split = np.cumsum([0] + [n*m for n, m in self.obs_split])
        self.obs_predator_index = slice(obs_split[2], obs_split[3])
        self.obs_prey_index = slice(obs_split[1], obs_split[2])

        num_obs = 6*self.num_boxes + 13*self.num_predators + 13*self.num_preys + 13
        ones = np.ones(num_obs)
        self.obs_space = spaces.Box(-ones*np.inf, ones*np.inf)
        self.state_space = self.obs_space

        def reset_actors(_, env_ids):
            randperm = torch.randperm(len(self.grid_avail), device=self.device)
            num_samples = len(env_ids)*self.num_agents
            sample_idx = randperm[torch.arange(num_samples).reshape(len(env_ids), self.num_agents)%len(self.grid_avail)]
            self.root_positions[env_ids, self.env_drone_slice] = self.grid_centers[sample_idx]
        
            # self.task_config["spawn_pos_idx"][env_ids] = sample_idx
        self.on_reset(reset_actors)

    def allocate_buffers(self):
        super().allocate_buffers()
        self.captured_steps_buf = torch.zeros(self.num_envs, self.num_preys, device=self.device)
        def reset_buffer(_, env_ids):
            self.captured_steps_buf[env_ids].zero_()
        
        self.on_reset(reset_buffer)
    
    def compute_state_and_obs(self, tensordict: TensorDictBase) -> TensorDictBase:
        obs_tensor = []
        identity = torch.eye(self.num_agents, device=self.device, dtype=bool)
        states_self = self.root_states[:, self.env_actor_index["drone"]]

        states_prey = self.root_states[:, self.env_prey_index].unsqueeze(1).repeat(1, self.num_agents, 1, 1)
        states_prey[..., :3] = states_prey[..., :3] - states_self[..., :3].unsqueeze(2)
        states_prey = states_prey.reshape(self.num_envs, self.num_agents, -1)

        states_predator = self.root_states[:, self.env_predator_index].unsqueeze(1).repeat(1, self.num_agents, 1, 1)
        states_predator[..., :3] = states_predator[..., :3] - states_self[..., :3].unsqueeze(2) # (env, agent, agent, 3) - (env, agent, 1, 3)
        states_predator = states_predator.reshape(self.num_envs, self.num_agents, -1)

        states_box = self.box_states.repeat(self.num_envs, self.num_agents, 1, 1)
        states_box[..., :3] = states_box[..., :3] - states_self[..., :3].unsqueeze(2)
        states_box = states_box.reshape(self.num_envs, self.num_agents, -1)

        assert not torch.isnan(states_self).any()
        obs_tensor.append(states_box)
        obs_tensor.append(states_prey)
        obs_tensor.append(states_predator)
        obs_tensor.append(states_self)
        
        obs_tensor = torch.cat(obs_tensor, dim=-1)
        return {
            "next_obs@predator": obs_tensor[:, self.predator_index],
            "next_obs@prey": obs_tensor[:, self.prey_index],
        }

    def compute_reward_and_done(self, tensordict: TensorDictBase) -> TensorDictBase:
        relative_pos = self.predator_pos - self.prey_pos
        distance = torch.norm(relative_pos, dim=-1, keepdim=True)
        distance_reward = 1.0 / (1.0 + distance ** 2) # (num_envs, num_predators)

        contact = self.contact_forces[:, self.env_body_index["base"]]

        collision_penalty = contact.any(-1).float()

        captured: torch.Tensor = distance.min(1).values < self.capture_radius
        self.captured_steps_buf[~captured] = 0
        self.captured_steps_buf[captured] += 1

        self.rew_buf[:, self.predator_index, 0] = distance_reward.mean(1)
        self.rew_buf[:, self.prey_index, 0] = -distance_reward.sum(1)

        self.rew_buf[..., 1] = -collision_penalty
        self.rew_buf[self.quadrotor_pos[..., 2]>self.MAX_XYZ[2], 1] -= 1
        self.rew_buf[..., 1] *= 2
        
        self.rew_buf[:, self.predator_index, 2] = captured.float()
        self.rew_buf[:, self.prey_index, 2] = -captured.float()
        
        self.cum_rew_buf.add_(self.rew_buf)
        
        self.reset_buf.zero_()
        # self.reset_buf[(distance > 3).all(-1)] = 1
        # self.reset_buf[self.quadrotor_pos[..., 2] < 0.1] = 1
        # self.reset_buf[self.quadrotor_pos[..., 2] > self.MAX_XYZ[2]] = 1
        self.reset_buf[self.progress_buf >= self.max_episode_length - 1] = 1

        cum_rew_buf = self.cum_rew_buf.clone()
        self.extras["episode"].update({
            "reward/distance@predator": cum_rew_buf[..., self.predator_index, 0],
            "reward/collision@predator": cum_rew_buf[..., self.predator_index, 1],
            "reward/capture@predator": cum_rew_buf[..., self.predator_index, 2],

            "reward/distance@prey": cum_rew_buf[..., self.prey_index, 0],
            "reward/collision@prey": cum_rew_buf[..., self.prey_index, 1],
            "reward/capture@prey": cum_rew_buf[..., self.prey_index, 2],

            "length": self.progress_buf + 1,
            "success": (self.captured_steps_buf > self.success_threshold).float(),
        })
        return {
            "reward@predator": self.rew_buf[:, self.predator_index],
            "reward@prey": self.rew_buf[:, self.prey_index],
            "done@predator": self.reset_buf[:, self.predator_index],
            "done@prey": self.reset_buf[:, self.prey_index],
            "env_done": self.reset_buf.all(-1),
        }

    def step(self, tensordict: TensorDictBase) -> TensorDictBase:
        tensordict["actions@predator"] = tensordict["actions@predator"].reshape(self.num_envs, self.num_predators, -1)
        tensordict["actions@prey"] = tensordict["actions@prey"].reshape(self.num_envs, self.num_preys, -1)

        actions = torch.cat([tensordict["actions@predator"], tensordict["actions@prey"]], dim=1)
        step_result = super().step({"actions": actions})
        del step_result["actions"]
        tensordict.update(step_result)
        return tensordict

    def pre_physics_step(self, tensordict: TensorDictBase):
        super().pre_physics_step(tensordict)
        if self.viewer:
            points = torch.cat([self.predator_pos[0].unsqueeze(0).expand(self.num_preys, -1, -1), self.prey_pos[0].unsqueeze(1).expand(-1, self.num_predators, -1)], dim=-1).flatten(0, 1).cpu().numpy()

            self.viewer_lines.append((points, [0.1, 0.2, 1]*len(points)))

    @property
    def predator_pos(self) -> torch.Tensor:
        return self.root_positions[:, self.env_predator_index]
    
    @property
    def prey_pos(self) -> torch.Tensor:
        return self.root_positions[:, self.env_prey_index]

    def get_dummy_policy(self, *args, **kwargs):
        def dummy_predator_policy(tensordict: TensorDict):
            prey_relative_pos = tensordict["obs@predator"][..., self.obs_prey_index].reshape(self.num_envs, self.num_predators, self.num_preys, 13)[..., :3]
            distance = torch.norm(prey_relative_pos, dim=3, keepdim=True)
            closest = torch.argmin(distance, dim=2, keepdim=True)

            tensordict["actions@predator"] = prey_relative_pos[:, :, 0] # torch.take_along_dim(prey_relative_pos, closest, dim=2)
            tensordict["action_log_prob@predator"] = torch.ones(self.num_envs, self.num_predators, 1)
            tensordict["value@predator"] = torch.zeros(self.num_envs, self.num_predators, 1)
            return tensordict

        def dummy_prey_policy(tensordict: TensorDict):
            predator_relative_pos = tensordict["obs@prey"][..., self.obs_predator_index].reshape(self.num_envs, self.num_preys, self.num_predators, 13)[..., :3]
            tensordict["actions@prey"] = -normalize(predator_relative_pos.mean(dim=-2))
            tensordict["action_log_prob@prey"] = torch.ones(self.num_envs, self.num_preys, 1)
            tensordict["value@prey"] = torch.zeros(self.num_envs, self.num_preys, 1)
            return tensordict
        
        def dummy_policy(tensordict: TensorDict):
            dummy_predator_policy(tensordict)
            dummy_prey_policy(tensordict)
            return tensordict
        
        agent = kwargs.get("agent")
        if agent == "predator":
            return dummy_predator_policy
        elif agent == "prey":
            return dummy_prey_policy
        elif agent is None:
            return dummy_policy        