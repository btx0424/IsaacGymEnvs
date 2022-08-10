import torch
import os
import math
import numpy as np

from isaacgym import gymapi, gymtorch, gymutil
from gym import spaces
from xml.etree import ElementTree
from typing import Callable, Dict, Tuple, Any
from torch import Tensor
from collections import defaultdict
from torchrl.data.tensordict.tensordict import TensorDictBase

from tqdm import tqdm
from .controller import DSLPIDControl
from ..base.vec_task import MultiAgentVecTask

from torchrl.data import TensorDict
from torchrl.envs.common import _EnvClass 

class QuadrotorBase(MultiAgentVecTask, _EnvClass):
    
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture: bool = False, force_render: bool = False):
        
        # no effect...
        cfg["env"]["numObservations"] = 13
        cfg["env"]["numActions"] = 4

        self.cfg = cfg
        
        self.actor_types = ["drone", "box", "sphere"]
        self.num_targets = cfg["env"].get("numTargets", 2)

        self.max_linear_velocity = cfg["env"].get("maxLinearVelocity", 4)
        
        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)

        self.root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim) 
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        
        # prepare tensors
        vec_root_tensor: Tensor = gymtorch.wrap_tensor(self.root_tensor)
        vec_root_tensor = vec_root_tensor.view(self.num_envs, self.actors_per_env, 13) # (num_envs, env_actor_count, 13)

        # TODO: replace it with the Omni Isaac gym api!
    
        self.root_states = vec_root_tensor
        self.root_positions = vec_root_tensor[..., 0:3]
        self.root_quats = vec_root_tensor[..., 3:7]
        self.root_linvels = vec_root_tensor[..., 7:10]
        self.root_angvels = vec_root_tensor[..., 10:13]
        self.contact_forces: Tensor = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, self.bodies_per_env, 3)
        
        self.refresh_tensors()
        self.initial_root_states = self.root_states.clone()

        self.forces = torch.zeros(
            (self.num_envs, self.bodies_per_env, 3), dtype=torch.float32, device=self.device, requires_grad=False)
        self.z_torques = torch.zeros(
            (self.num_envs, self.bodies_per_env, 3), dtype=torch.float32, device=self.device, requires_grad=False)
        self.max_episode_length = self.cfg["env"]["maxEpisodeLength"]

        if self.viewer:
            cam_pos = gymapi.Vec3(-2.3, 0, 4.2)
            cam_target = gymapi.Vec3(0, 0, 0.1)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        self.controller = DSLPIDControl(n=self.num_envs * self.num_agents, sim_params=self.sim_params, kf=self.KF, device=self.device)
        
        self.act_type = cfg["env"].get("actType", "pid_vel")
        self.obs_type = cfg["env"].get("obsType", "root_state")
        self.create_act_space_and_processor(self.act_type)
        self.create_obs_space_and_processor(self.obs_type)

    def create_sim(self):
        # create sim
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81

        self.sim = self.gym.create_sim(
            self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        assert self.sim is not None
        
        # boundary boxes
        spacing = self.cfg["env"]["envSpacing"]
        self.MAX_XYZ = torch.tensor([spacing, spacing, 2], device=self.device)
        self.MIN_XYZ = torch.tensor([-spacing, -spacing, 0], device=self.device)
        _wall_length = 2 * spacing
        _wall_height = self.MAX_XYZ[2]
        _walls = torch.tensor([
            [-spacing, 0, 0.5, 0.05, _wall_length, _wall_height],
            [0, -spacing, 0.5, _wall_length, 0.05, _wall_height],
            [spacing, 0, 0.5, 0.05, _wall_length, _wall_height],
            [0, spacing, 0.5, _wall_length, 0.05, _wall_height]], device=self.device)
        _obstacles = torch.tensor([
            [1, 1, 0.3, 0.25, 0.25, _wall_height],
            [-1, -1, 0.3, 0.25, 0.25, _wall_height]], device=self.device)

        self.boxes = []
        self.boxes.append(_walls)
        self.boxes.append(_obstacles)
        self.box_states = torch.cat(self.boxes, dim=0)
        self.num_boxes = len(self.box_states)
        
        # load assets
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../assets')
        asset_file = "urdf/quadrotor/cf2x.urdf"
        asset_urdf_tree = ElementTree.parse(os.path.join(asset_root, asset_file)).getroot()
        self.KF = float(asset_urdf_tree[0].attrib['kf'])
        self.KM = float(asset_urdf_tree[0].attrib['km'])
        self.THRUST2WEIGHT_RATIO = float(asset_urdf_tree[0].attrib['thrust2weight'])
        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0
        asset_options.linear_damping = 0
        asset_options.max_linear_velocity = self.max_linear_velocity

        self.HOVER_RPM = math.sqrt(9.81 * 0.027 / (4*self.KF))
        self.MAX_RPM = np.sqrt((self.THRUST2WEIGHT_RATIO*9.81) / (4*self.KF))
        self.TIME_STEP = self.sim_params.dt
        self.dt = self.sim_params.dt
        asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        box_assets = {tuple(half_ext.tolist()): self.gym.create_box(self.sim, *half_ext, asset_options)
            for half_ext in set([box[3:] for box in self.box_states])}
        asset_options.fix_base_link = False
        asset_options.disable_gravity = True
        sphere_asset = self.gym.create_sphere(self.sim, 0.03, asset_options)

        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

        self.envs = []
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)
        num_per_row = int(math.sqrt(self.num_envs))

        drone_pose = gymapi.Transform()
        box_pose = gymapi.Transform()
        sphere_pose = gymapi.Transform(p=gymapi.Vec3(0.0, 0.0, -1.))

        env_actor_index = defaultdict(lambda:[])
        env_body_index = defaultdict(lambda:[])
        sim_actor_index = defaultdict(lambda:[])

        cell_size = 0.4
        grid_shape = ((self.MAX_XYZ - self.MIN_XYZ) / cell_size).int()
        centers = torch.tensor(list(np.ndindex(*(grid_shape.cpu()))), device=self.rl_device) + 0.5
        avail = torch.ones(len(centers), dtype=bool)

        for i_env in tqdm(range(self.num_envs), desc="Creating envs"):
            env = self.gym.create_env(self.sim, lower, upper, num_per_row)
            drone_pose.p = gymapi.Vec3(0.0, 0.0, 0.25)

            for i_agent in range(self.num_agents):
                drone_handle = self.gym.create_actor(env, asset, drone_pose, f"cf2x_{i_agent}", i_env, 0)
                drone_pose.p.x += 0.2
                drone_pose.p.y += 0.2
                drone_pose.p.z += 0.15
                if i_env == 0:
                    # actor index
                    env_actor_index["drone"].append(
                        self.gym.get_actor_index(env, drone_handle, gymapi.DOMAIN_ENV))
                    # body index
                    env_body_index["prop"].extend([
                        self.gym.get_actor_rigid_body_index(env, drone_handle, body_index, gymapi.DOMAIN_ENV) 
                        for body_index in [2, 3, 4, 5]])
                    env_body_index["base"].append(
                        self.gym.get_actor_rigid_body_index(env, drone_handle, 0, gymapi.DOMAIN_ENV))
                sim_actor_index["drone"].append(self.gym.get_actor_index(env, drone_handle, gymapi.DOMAIN_SIM))

            for i_target in range(self.num_targets):
                sphere_handle = self.gym.create_actor(env, sphere_asset, sphere_pose, f"sphere_{i_target}", i_env, 1)
                if i_env == 0:
                    env_actor_index["target"].append(
                        self.gym.get_actor_index(env, sphere_handle, gymapi.DOMAIN_ENV))
                sim_actor_index["target"].append(self.gym.get_actor_index(env, sphere_handle, gymapi.DOMAIN_SIM))

            for i_box in range(self.num_boxes):
                center, half_ext = self.box_states[i_box][:3], self.box_states[i_box][3:]
                box_asset = box_assets[tuple(half_ext.tolist())]
                box_pose.p = gymapi.Vec3(*center)
                box_handle = self.gym.create_actor(env, box_asset, box_pose, f"box_{i_box}", i_env, 0) 
                
                if i_env == 0:
                    min_corner = torch.floor((center-half_ext-self.MIN_XYZ) / cell_size)
                    max_corner = torch.ceil((center+half_ext-self.MIN_XYZ) / cell_size)
                    mask = (centers > min_corner).all(1) & (centers < max_corner).all(1)
                    avail[mask] = False

                    env_actor_index["box"].append(
                        self.gym.get_actor_index(env, box_handle, gymapi.DOMAIN_ENV))
                sim_actor_index["box"].append(self.gym.get_actor_index(env, box_handle, gymapi.DOMAIN_SIM))
            self.envs.append(env)

        self.env_actor_index = {name: torch.tensor(index) for name, index in env_actor_index.items()}
        self.env_body_index = {name: torch.tensor(index) for name, index in env_body_index.items()}
        self.bodies_per_env = self.gym.get_env_rigid_body_count(env)
        self.actors_per_env = self.gym.get_actor_count(env)
        self.sim_actor_index = {name: torch.tensor(index, dtype=torch.int32, device=self.device).reshape(self.num_envs, -1) for name, index in sim_actor_index.items()}
        self.sim_actor_index["__all__"] = torch.cat(list(self.sim_actor_index.values()), dim=1)
        assert self.sim_actor_index["__all__"].size(-1) == self.actors_per_env

        self.grid_centers = centers * cell_size + self.MIN_XYZ
        self.grid_avail = torch.nonzero(avail).flatten()
        
    def reset_idx(self, env_ids):
        self.controller.reset_idx(env_ids, self.num_envs)

        self.reset_buffers(env_ids)
        self.reset_actors(env_ids)
        root_reset_ids = self.sim_actor_index["__all__"][env_ids].flatten()
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, self.root_tensor, gymtorch.unwrap_tensor(root_reset_ids), len(root_reset_ids))
    
    def reset_buffers(self, env_ids):
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0
        self.cum_rew_buf[env_ids] = 0

    def reset_actors(self, env_ids):
        self.root_states[env_ids] = self.initial_root_states[env_ids]

    def pre_physics_step(self, actions: torch.Tensor):
        actions = actions.view(self.num_envs, self.num_agents, 4)
        reset_env_ids = self.reset_buf.all(-1).nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        rpms = torch.clamp(actions, 0, self.MAX_RPM)
        forces = rpms**2 * self.KF # (env, actor, 4)
        torques = rpms**2 * self.KM
        z_torques = (-torques[..., 0] + torques[..., 1] - torques[..., 2] + torques[..., 3]) # (env, actor)

        self.forces[..., self.env_body_index["prop"], 2] = forces.reshape(self.num_envs, 4*self.num_agents)
        self.z_torques[..., self.env_body_index["base"], 2] = z_torques.reshape(self.num_envs, self.num_agents)

        self.gym.apply_rigid_body_force_tensors(self.sim, 
            gymtorch.unwrap_tensor(self.forces), 
            gymtorch.unwrap_tensor(self.z_torques), gymapi.LOCAL_SPACE)

    def post_physics_step(self):
        self.progress_buf += 1

        self.refresh_tensors()
        # self.compute_observations()
        self.compute_reward_and_reset()
        
        env_dones = self.reset_buf.all(-1)
        self.extras["env_dones"] = env_dones
        self.extras["episode"].update({
            "length": self.progress_buf.clone(),
        })

        if self.viewer and self.viewer_lines:
            self.gym.clear_lines(self.viewer)
            for points, color in self.viewer_lines:
                self.gym.add_lines(self.viewer, self.envs[0], len(points), points, color)
            self.viewer_lines.clear()
            
    # def compute_observations(self):
    #     self.obs_buf[..., :3] = self.quadrotor_pos / 3
    #     self.obs_buf[..., 3:7] = self.root_quats[:, self.env_actor_index["drone"]]
    #     self.obs_buf[..., 7:10] = self.root_linvels[:, self.env_actor_index["drone"]] / 2.
    #     self.obs_buf[..., 10:13] = self.root_angvels[:, self.env_actor_index["drone"]] / math.pi
    #     return self.obs_buf

    def compute_reward_and_reset(self):
        raise NotImplementedError

    def step(self, actions: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        actions = actions.view(self.num_envs, self.num_agents, -1)
        actions = self.act_processor(actions)

        obs_dict, reward, done, info = super().step(actions)
        obs_dict = TensorDict(obs_dict, batch_size=[self.num_envs, self.num_agents])
        self.obs_processor(obs_dict)
        
        return obs_dict, reward, done, info

    def reset(self, tensordict: TensorDict=None, **kwargs) -> TensorDictBase:
        tensordict_reset = TensorDict({}, batch_size=(self.num_envs, self.num_agents))
        self.obs_processor(tensordict_reset)
        self.extras["episode"] = TensorDict({}, batch_size=(self.num_envs,))
        self.viewer_lines = []
        self.reset_idx(torch.arange(self.num_envs))
        return tensordict_reset

    def refresh_tensors(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

    @property
    def quadrotor_states(self) -> TensorDict:
        pos, quat, linvel, angvel = self.root_states[:, self.env_actor_index["drone"]]\
            .split_with_sizes((3, 4, 3, 3), dim=-1)
        return TensorDict(
            {"pos":pos, "quat":quat, "linvel":linvel, "angvel":angvel}, (self.num_envs, self.num_agents))

    @property
    def quadrotor_pos(self) -> Tensor:
        return self.root_positions[:, self.env_actor_index["drone"]]

    @quadrotor_pos.setter
    def quadrotor_pos(self, pos: Tensor):
        self.root_positions[:, self.env_actor_index["drone"]] = pos

    @property
    def target_pos(self) -> Tensor:
        return self.root_positions[:, self.env_actor_index["target"]]

    @target_pos.setter
    def target_pos(self, pos: Tensor):
        self.root_positions[:, self.env_actor_index["target"]] = pos
    
    @property
    def target_vel(self) -> Tensor:
        return self.root_linvels[:, self.env_actor_index["target"]]

    @target_vel.setter
    def target_vel(self, vel: Tensor):
        self.root_linvels[:, self.env_actor_index["target"]] = vel

    def __repr__(self) -> str:
        obs_space = f"obs_space: {self.observation_space}"
        act_space = f"act_space: {self.action_space}"
        env_actor_index = "\n".join([f"{k}_index: {v}" for k, v in self.env_actor_index.items()])

        return "\n".join([obs_space, act_space, env_actor_index])
    
    def create_obs_space_and_processor(self, obs_type=None) -> None:
        raise NotImplementedError
    
    def create_act_space_and_processor(self, act_type) -> None:
        if act_type == "pid_pos":
            ones = np.ones(3)
            self.act_space = spaces.Box(-ones, ones)
            def act_processor(actions: Tensor) -> Tensor:
                pos, quat, vel, angvel = self.quadrotor_states.reshape(-1).values()
                target_pos = pos + actions.reshape(-1, 3)
                if self.viewer:
                    points = torch.cat([pos[:self.num_agents], target_pos[:self.num_agents]], dim=1).cpu().numpy()
                    self.viewer_lines.append(points, [[1., 0., 0.]]*len(points))
                rpms = self.controller.compute_control(
                    self.TIME_STEP,
                    pos, quat, vel, angvel,
                    target_pos,
                    torch.zeros((self.num_envs*self.num_agents, 3), device=self.device),
                    torch.zeros((self.num_envs*self.num_agents, 3), device=self.device),
                    torch.zeros((self.num_envs*self.num_agents, 3), device=self.device),
                )[0]
                return rpms.view(self.num_envs, self.num_agents, 4)
            self.act_processor = act_processor
        elif act_type == "pid_vel":
            ones = np.ones(3)
            self.act_space = spaces.Box(-ones * 3, ones * 3)
            def act_processor(actions: Tensor) -> Tensor:
                pos, quat, vel, angvel = self.quadrotor_states.reshape(-1).values()
                target_vel = actions.reshape(-1, 3)
                if self.viewer:
                    points = torch.cat([pos[:self.num_agents], pos[:self.num_agents] + target_vel[:self.num_agents]], dim=1).cpu().numpy()
                    self.viewer_lines.append((points, [[1., 0., 0.]]*len(points)))
                rpms = self.controller.compute_control(
                    self.TIME_STEP,
                    pos, quat, vel, angvel,
                    pos,
                    torch.zeros((self.num_envs*self.num_agents, 3), device=self.device),
                    target_vel,
                    torch.zeros((self.num_envs*self.num_agents, 3), device=self.device),
                )[0]
                return rpms.view(self.num_envs, self.num_agents, 4)
            self.act_processor = act_processor
        elif act_type == "multi_discrete":
            self.act_space = spaces.Tuple([spaces.Discrete(3)] * 3)
            self.act_space = spaces.MultiDiscrete([3, 3, 3])
            def act_processor(actions: Tensor) -> Tensor:
                pos, quat, vel, angvel = self.quadrotor_states.reshape(-1).values()
                target_vel = (actions.reshape(-1, 3) - 1)
                if self.viewer:
                    points = torch.cat([pos[:self.num_agents], pos[:self.num_agents] + target_vel[:self.num_agents]], dim=1).cpu().numpy()
                    self.viewer_lines.append((points, [[1., 0., 0.]]*len(points)))
                rpms = self.controller.compute_control(
                    self.TIME_STEP,
                    pos, quat, vel, angvel,
                    pos,
                    torch.zeros((self.num_envs*self.num_agents, 3), device=self.device),
                    target_vel,
                    torch.zeros((self.num_envs*self.num_agents, 3), device=self.device),
                )[0]
                return rpms.view(self.num_envs, self.num_agents, 4)
            self.act_processor = act_processor
        elif act_type == "rpm":
            ones = np.ones(4)
            self.thrusts = torch.ones(self.num_envs, self.num_agents, 4, device=self.device) * self.HOVER_RPM
            self.act_space = spaces.Box(-ones, ones)
            def act_processor(actions: Tensor) -> Tensor:
                self.thrusts.add_(self.dt * actions * 20000)
                self.thrusts.clamp_max_(self.HOVER_RPM * 1.5)
                return self.thrusts
            self.act_processor = act_processor
        else:
            raise NotImplementedError(act_type)
    