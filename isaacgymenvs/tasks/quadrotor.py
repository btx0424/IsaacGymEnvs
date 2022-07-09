from collections import defaultdict

from .base.vec_task import MultiAgentVecTask
from isaacgym import gymapi, gymtorch, gymutil

import torch
import os
import math
import numpy as np
from gym import spaces
from xml.etree import ElementTree
from isaacgymenvs.utils.torch_jit_utils import *
from typing import Dict, Tuple
from torch import Tensor

class Quadrotor(MultiAgentVecTask):
    
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture: bool = False, force_render: bool = False):
        
        cfg["env"]["numObservations"] = 13
        cfg["env"]["numActions"] = 4
        
        self.cfg = cfg
        self.actor_types = ["drone", "box"]

        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)

        self.root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim) 
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.refresh_tensors()

        vec_root_tensor: Tensor = gymtorch.wrap_tensor(self.root_tensor)
        vec_root_tensor = vec_root_tensor.view(self.num_envs, self.actors_per_env, 13) # (num_envs, env_actor_count, 13)
        
        print("root_tensor shape:", vec_root_tensor.shape)
        for name, index in self.env_actor_index.items():
            print(name, index)
        for name, index in self.env_body_index.items():
            print(name, index)

        # TODO: replace it with the Omni Isaac gym api
        
        self.root_states = vec_root_tensor
        self.root_positions = vec_root_tensor[..., 0:3]
        self.root_quats = vec_root_tensor[..., 3:7]
        self.root_linvels = vec_root_tensor[..., 7:10]
        self.root_angvels = vec_root_tensor[..., 10:13]
        self.contact_forces: Tensor = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, self.bodies_per_env, 3)
        
        self.initial_root_states = self.root_states.clone()
        self.targets = self.root_positions[:, self.env_actor_index["drone"]].clone()
        self.targets[..., 2] += 0.5

        self.forces = torch.zeros(
            (self.num_envs, self.bodies_per_env, 3), dtype=torch.float32, device=self.device, requires_grad=False)
        self.z_torques = torch.zeros(
            (self.num_envs, self.bodies_per_env, 3), dtype=torch.float32, device=self.device, requires_grad=False)
        self.max_episode_length = self.cfg["env"]["maxEpisodeLength"]

        if self.viewer:
            cam_pos = gymapi.Vec3(1.0, 1.0, 1.8)
            cam_target = gymapi.Vec3(2.2, 2.0, 1.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        ones = np.ones(self.num_obs)
        self.obs_space = spaces.Box(-ones*np.inf, ones*np.inf)
        ones = np.ones(3)
        self.act_space = spaces.Box(-ones, ones)

    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81

        self.sim = self.gym.create_sim(
            self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        assert self.sim is not None
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets')
        asset_file = "urdf/quadrotor/cf2x.urdf"
        asset_urdf_tree = ElementTree.parse(os.path.join(asset_root, asset_file)).getroot()
        self.KF = float(asset_urdf_tree[0].attrib['kf'])
        self.KM = float(asset_urdf_tree[0].attrib['km'])
        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0
        asset_options.linear_damping = 0

        self.HOVER_RPM = math.sqrt(9.81 / (4*self.KF))
        self.TIME_STEP = self.sim_params.dt
        asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        box_asset = self.gym.create_box(self.sim, 0.1, 0.1, 1., asset_options)

        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

        self.envs = []
        spacing = self.cfg["env"]["envSpacing"]
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)
        num_per_row = int(math.sqrt(self.num_envs))

        drone_pose = gymapi.Transform()
        box_pose = gymapi.Transform()

        env_actor_index = defaultdict(lambda:[])
        env_body_index = defaultdict(lambda:[])
        sim_root_index = []

        for i_env in range(self.num_envs):
            env = self.gym.create_env(self.sim, lower, upper, num_per_row)
            drone_pose.p = gymapi.Vec3(0.0, 0.0, 0.25)

            for i_agent in range(self.num_agents):
                drone_handle = self.gym.create_actor(env, asset, drone_pose, f"cf2x_{i_agent}", i_env, 0)
                drone_pose.p.x += 0.1
                drone_pose.p.z += 0.1
                if i_env == 0:
                    env_actor_index["drone"].append(
                        self.gym.get_actor_index(env, drone_handle, gymapi.DOMAIN_ENV))
                    env_body_index["prop"].extend([
                        self.gym.get_actor_rigid_body_index(env, drone_handle, body_index, gymapi.DOMAIN_ENV) 
                        for body_index in [2, 3, 4, 5]])
                    env_body_index["base"].append(
                        self.gym.get_actor_rigid_body_index(env, drone_handle, 0, gymapi.DOMAIN_ENV))
                sim_root_index.append(self.gym.get_actor_index(env, drone_handle, gymapi.DOMAIN_SIM))

            for i_box, center in enumerate([
                    [-1, 1, 0.5],
                    [1, 1, 0.5],
                    [1, -1, 0.5],
                    [-1, -1, 0.5]]):
                box_pose.p = gymapi.Vec3(*center)
                box_handle = self.gym.create_actor(env, box_asset, box_pose, f"box_{i_box}", i_env, 0) 
                if i_env == 0:
                    env_actor_index["box"].append(
                        self.gym.get_actor_index(env, box_handle, gymapi.DOMAIN_ENV))
                sim_root_index.append(self.gym.get_actor_index(env, box_handle, gymapi.DOMAIN_SIM))
            self.envs.append(env)

        self.env_actor_index = {name: torch.tensor(index) for name, index in env_actor_index.items()}
        self.env_body_index = {name: torch.tensor(index) for name, index in env_body_index.items()}
        self.bodies_per_env = self.gym.get_env_rigid_body_count(env)
        self.actors_per_env = self.gym.get_actor_count(env)
        self.sim_root_index = torch.tensor(sim_root_index, dtype=torch.int32, device=self.device).reshape(self.num_envs, self.actors_per_env)

        self.controller = DSLPIDControl(n=self.num_envs * self.num_agents, sim_params=self.sim_params, kf=self.KF, device=self.device)

    def reset_idx(self, env_ids):
        self.root_states[env_ids] = self.initial_root_states[env_ids]
        pos = self.root_positions[:, self.env_actor_index["drone"]]
        # print(pos.shape)
        # self.root_positions[:, self.env_actor_index["drone"]] += torch.rand_like(pos) * 0.5
        reset_indices = self.sim_root_index[env_ids].flatten()
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, self.root_tensor, gymtorch.unwrap_tensor(reset_indices), len(reset_indices))

        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0
        
        self.controller.reset_idx(env_ids, self.num_envs) # TODO: fix reset

    def pre_physics_step(self, actions: torch.Tensor):
        actions = actions.view(self.num_envs, self.num_agents, 4)
        reset_env_ids = self.reset_buf.all(-1).nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)
        actions = actions.to(self.device)

        # rpms = self.HOVER_RPM * (1+0.05*actions)
        rpms = actions
        forces = rpms**2 * self.KF # (env, actor, 4)
        torques = rpms**2 * self.KM
        z_torques = (-torques[..., 0] + torques[..., 1] - torques[..., 2] + torques[..., 3]) # (env, actor)

        self.forces[..., self.env_body_index["prop"], 2] = forces.flatten(start_dim=1)
        self.z_torques[..., self.env_body_index["base"], 2] = z_torques.flatten(start_dim=1)
        self.forces[reset_env_ids] = 0
        self.z_torques[reset_env_ids] = 0

        self.gym.apply_rigid_body_force_tensors(self.sim, 
            gymtorch.unwrap_tensor(self.forces), 
            gymtorch.unwrap_tensor(self.z_torques), gymapi.LOCAL_SPACE)

    def post_physics_step(self):
        self.progress_buf += 1
        self.refresh_tensors()
        self.compute_observations()
        self.compute_reward()
    
    def compute_observations(self):
        target = torch.tensor([0., 0., 1.], device=self.device)
        self.obs_buf[..., :3] = (target - self.root_positions[:, self.env_actor_index["drone"]]) / 3
        self.obs_buf[..., 3:7] = self.root_quats[:, self.env_actor_index["drone"]]
        self.obs_buf[..., 7:10] = self.root_linvels[:, self.env_actor_index["drone"]] / 2.
        self.obs_buf[..., 10:13] = self.root_angvels[:, self.env_actor_index["drone"]] / math.pi
        return self.obs_buf

    def compute_reward(self):
        pos, quat, vel, angvel = self.root_states[:, self.env_actor_index["drone"]] \
            .reshape(self.num_envs, self.num_agents, 13) \
            .split_with_sizes((3, 4, 3, 3), dim=-1)
        contact = self.contact_forces[:, self.env_body_index["base"]]
        self.rew_buf[:], self.reset_buf[:] = compute_quadcopter_reward(
            pos, quat, vel, angvel, contact, self.targets,
            self.reset_buf, self.progress_buf, self.max_episode_length
        )
        
    def step(self, actions: torch.Tensor):
        target_pos = actions
        pos, quat, vel, angvel = self.root_states[:, self.env_actor_index["drone"]] \
            .reshape(-1, 13) \
            .split_with_sizes((3, 4, 3, 3), dim=-1)
        actions = self.controller.compute_control(
                self.TIME_STEP,
                pos, quat, vel, angvel,
                target_pos.reshape(-1, 3),
                torch.zeros((self.num_envs*self.num_agents, 3), device=self.device),
                torch.zeros((self.num_envs*self.num_agents, 3), device=self.device),
                torch.zeros((self.num_envs*self.num_agents, 3), device=self.device),
        )[0].view(self.num_envs, self.num_agents, 4)
        obs_dict, reward, done, info = super().step(actions)
        return obs_dict["obs"], reward.clone(), done, info
    
    def reset(self):
        self.controller.reset()
        obs_dict = super().reset()
        return obs_dict["obs"]

    def refresh_tensors(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

# @torch.jit.script
def compute_quadcopter_reward(
        root_positions: Tensor, 
        root_quats: Tensor, 
        root_linvels: Tensor, 
        root_angvels: Tensor, 
        contact_forces: Tensor,
        targets: Tensor,
        reset_buf: Tensor, 
        progress_buf: Tensor, 
        max_episode_length: float) -> Tuple[Tensor, Tensor]:

    # distance to target
    target_dist = torch.norm(targets - root_positions, dim=-1)
    pos_reward = 1.0 / (1.0 + target_dist * target_dist)

    # uprightness
    # ups = quat_axis(root_quats.squeeze(), 2)
    # tiltage = torch.abs(1 - ups[..., 2])
    # up_reward = (1.0 / (1.0 + tiltage * tiltage)).unsqueeze(1)

    # spinning
    spinnage = torch.abs(root_angvels[..., 2])
    spinnage_reward = 1.0 / (1.0 + spinnage * spinnage)

    # collision
    collision = contact_forces.any(-1).float()
    
    # combined reward
    # uprigness and spinning only matter when close to the target
    reward = pos_reward + pos_reward * (spinnage_reward) - collision
    # resets due to misbehavior
    reset = torch.zeros_like(reset_buf)
    reset[target_dist > 3] = 1
    reset[root_positions[..., 2] < 0.1] = 1

    # resets due to episode length
    reset[progress_buf >= max_episode_length - 1] = 1
    return reward, reset

from .utils import *

class DSLPIDControl:
    """
    TODO: @ Botian: completely test the functionality of this controller...
    """
    def __init__(self, n, sim_params: gymapi.SimParams, kf, device="cpu") -> None:
        self.device = device
        self.P_COEFF_FOR = torch.tensor([.4, .4, 1.25], device=device)
        self.I_COEFF_FOR = torch.tensor([.05, .05, .05], device=device)
        self.D_COEFF_FOR = torch.tensor([.2, .2, .5], device=device)
        self.P_COEFF_TOR = torch.tensor([70000., 70000., 60000.], device=device)
        self.I_COEFF_TOR = torch.tensor([.0, .0, 500.], device=device)
        self.D_COEFF_TOR = torch.tensor([20000., 20000., 12000.], device=device)
        self.PWM2RPM_SCALE = 0.2685
        self.PWM2RPM_CONST = 4070.3
        self.MIN_PWM = 20000
        self.MAX_PWM = 65535
        self.GRAVITY = abs(sim_params.gravity.z) * 0.027
        self.MIXER_MATRIX = torch.tensor([ [.5, -.5,  -1], [.5, .5, 1], [-.5,  .5,  -1], [-.5, -.5, 1] ], device=device)
        self.KF = kf
        self.N = n

    def reset(self):
        n = self.N
        self.last_pos_e = torch.zeros((n, 3), device=self.device)
        self.integral_pos_e = torch.zeros((n, 3), device=self.device)
        self.last_rpy = torch.zeros((n, 3), device=self.device)
        self.last_rpy_e = torch.zeros((n, 3), device=self.device)
        self.integral_rpy_e = torch.zeros((n, 3), device=self.device)

    def reset_idx(self, ids: torch.Tensor, num_envs: int):
        self.last_pos_e.view(num_envs, -1)[ids] = 0
        self.integral_pos_e.view(num_envs, -1)[ids] = 0
        self.last_rpy.view(num_envs, -1)[ids] = 0
        self.last_rpy_e.view(num_envs, -1)[ids] = 0
        self.integral_rpy_e.view(num_envs, -1)[ids] = 0

    def compute_control(self, control_timestep, 
            cur_pos, cur_quat, cur_vel, cur_ang_vel,
            target_pos, target_rpy, target_vel, target_rpy_rates):
        thrust, computed_target_rotation, pos_e = self._position_control(
            control_timestep,
            cur_pos,
            cur_quat,
            cur_vel,
            target_pos,
            target_rpy,
            target_vel
            )
        rpm = self._attitude_control(
            control_timestep,
            thrust,
            cur_quat,
            computed_target_rotation,
            target_rpy_rates
            )
        return rpm, pos_e

    def _position_control(self, 
            control_timestep: float,
            cur_pos: torch.Tensor,
            cur_quat: torch.Tensor,
            cur_vel: torch.Tensor,
            target_pos: torch.Tensor,
            target_rpy: torch.Tensor,
            target_vel: torch.Tensor):
        
        pos_e = target_pos - cur_pos
        vel_e = target_vel - cur_vel
        cur_rotation = quaternion_to_rotation_matrix(cur_quat) # (*, 3, 3)
        self.integral_pos_e = self.integral_pos_e + pos_e*control_timestep
        self.integral_pos_e = torch.clip(self.integral_pos_e, -2., 2.)
        self.integral_pos_e[..., 2] = torch.clip(self.integral_pos_e[..., 2], -0.15, .15)

        target_thrust = self.P_COEFF_FOR * pos_e \
            + self.I_COEFF_FOR * self.integral_pos_e \
            + self.D_COEFF_FOR * vel_e \
            + torch.tensor([0, 0, self.GRAVITY], device=self.device) # (*, 3)
        scalar_thrust = torch.relu(target_thrust.view(-1, 1, 3) @ cur_rotation[..., :, 2].view(-1, 3, 1)).view(-1, 1)
        thrust = (torch.sqrt(scalar_thrust / (4*self.KF)) - self.PWM2RPM_CONST) / self.PWM2RPM_SCALE
        target_z_ax = target_thrust / torch.linalg.norm(target_thrust, axis=-1, keepdims=True)
        target_x_c = torch.stack([
            torch.cos(target_rpy[..., 2]), torch.sin(target_rpy[..., 2]), torch.zeros_like(target_rpy[..., 2])], dim=-1)
        target_y_ax = torch.cross(target_z_ax, target_x_c) / torch.linalg.norm(torch.cross(target_z_ax, target_x_c), axis=-1, keepdims=True)
        target_x_ax = torch.cross(target_y_ax, target_z_ax)
        target_rotation = torch.stack([target_x_ax, target_y_ax, target_z_ax], dim=-1)
        # print("target_thrust", target_thrust)
        # print("scalar_thrust", scalar_thrust)
        # print("thrust", thrust)
        # print("target_z_ax", target_z_ax)
        # print("target_x_c", target_x_c)
        # print("target_y_ax", target_y_ax)
        # print("target_x_ax", target_x_ax)
        # print("target_rotation", target_rotation)
        # raise
        return thrust, target_rotation, pos_e
    
    def _attitude_control(self,
            control_timestep: float,
            thrust: torch.Tensor,
            cur_quat: torch.Tensor,
            target_rotation: torch.Tensor,
            target_rpy_rates: torch.Tensor):
        cur_rotation = quaternion_to_rotation_matrix(cur_quat)
        cur_rpy = quaternion_to_euler(cur_quat)

        rot_matrix_e = target_rotation.transpose(-1, -2) @ cur_rotation - cur_rotation.transpose(-1, -2) @ target_rotation

        rot_e = torch.stack([rot_matrix_e[..., 2, 1], rot_matrix_e[..., 0, 2], rot_matrix_e[..., 1, 0]], dim=-1)
        rpy_rates_e = target_rpy_rates - (cur_rpy - self.last_rpy) / control_timestep
        self.last_rpy = cur_rpy
        self.integral_rpy_e = self.integral_rpy_e - rot_e*control_timestep
        self.integral_rpy_e = torch.clip(self.integral_rpy_e, -1500., 1500.)
        self.integral_rpy_e[..., 0:2] = torch.clip(self.integral_rpy_e[..., 0:2], -1., 1.)
        #### PID target torques ####################################
        target_torques = - self.P_COEFF_TOR * rot_e \
                         + self.D_COEFF_TOR * rpy_rates_e \
                         + self.I_COEFF_TOR * self.integral_rpy_e
        # print("P * rot_e", self.P_COEFF_TOR * rot_e)
        # print("D * rpy_rates_e", self.D_COEFF_TOR * rpy_rates_e)
        # print("I * self.integral_rpy_e", self.I_COEFF_TOR * self.integral_rpy_e)
        
        target_torques = torch.clip(target_torques, -3200, 3200)
        pwm = thrust + (self.MIXER_MATRIX @ target_torques.T).T

        # print("cur_rotation: ", cur_rotation)
        # print("target_rotation: ", target_rotation)
        # print("rot_matrix_e: ", rot_matrix_e)
        # print("rot_e: ", rot_e)
        # print("rpy_rates_e: ", rpy_rates_e)
        # print("integral_rpy_e: ", self.integral_rpy_e)
        # print("target_torques: ", target_torques)
        # print("pwm: ", pwm)
        # print("thrust: ", thrust)
        # raise
        pwm = torch.clip(pwm, self.MIN_PWM, self.MAX_PWM)
        return self.PWM2RPM_SCALE * pwm + self.PWM2RPM_CONST
