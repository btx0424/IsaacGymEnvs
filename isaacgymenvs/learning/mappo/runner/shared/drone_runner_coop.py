from cmath import inf
from collections import defaultdict
from dataclasses import dataclass
import time
from typing import Any, Dict, Tuple
from isaacgymenvs.tasks.base.vec_task import MultiAgentVecTask
from isaacgymenvs.tasks.quadrotor import QuadrotorBase
import numpy as np
import torch
import wandb
from .base_runner import Runner
from tqdm import tqdm

from isaacgymenvs.learning.mappo.algorithms.rmappo import MAPPOPolicy
from isaacgymenvs.learning.mappo.utils.shared_buffer import SharedReplayBuffer


@dataclass
class RunnerConfig:
    num_envs: int
    num_agents: int = 1

    max_iterations: int = 500
    max_env_steps: int = int(5e6)

    log_interval: int = 5
    save_interval: int = 10


@dataclass
class PPOConfig:
    num_steps: int = 16
    num_mini_batch: int = 8


class DroneRunner(Runner):
    def __init__(self, config):
        cfg: RunnerConfig = config["cfg"]
        self.max_iterations = cfg.max_iterations
        self.max_env_steps = cfg.max_env_steps
        self.log_interval = cfg.log_interval

        self.num_envs = cfg.num_envs
        self.num_agents = cfg.num_agents

        all_args = config["all_args"]

        if all_args.use_attn:
            envs: QuadrotorBase = config["envs"]
            obs_split = envs.obs_split
            envs.share_observation_space = envs.obs_space = \
                [[sum(num*dim for num, dim in obs_split), *obs_split]]

        super().__init__(config)

        self.num_steps = all_args.num_steps

        envs: MultiAgentVecTask = config["envs"]

        self.policy = MAPPOPolicy(
            self.all_args,
            envs.obs_space,
            envs.state_space,
            envs.act_space)

        self.buffer = SharedReplayBuffer(
            self.all_args,
            self.num_agents,
            envs.obs_space,
            envs.state_space,
            envs.act_space)

    def run(self):
        obs = self.envs.reset()["obs"]
        obs = obs.reshape(self.num_envs, self.num_agents, *obs.shape[1:])
        share_obs = obs

        self.buffer.share_obs[0] = share_obs
        self.buffer.obs[0] = obs

        start = time.perf_counter()

        env_step_time = 0
        inf_step_time = 0

        total_env_steps = 0
        total_episodes = 0

        episode_infos = defaultdict(lambda: [])

        for iteration in range(self.max_iterations):

            if self.use_linear_lr_decay:
                self.policy.lr_decay(iteration, self.max_iterations)

            for step in range(self.num_steps):
                # Sample actions
                _step_start = time.perf_counter()
                values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect(
                    step)
                _inf_end = time.perf_counter()
                obs_dict, rewards, dones, infos = self.envs.step(actions)
                _env_end = time.perf_counter()

                # for compatibility
                obs = obs_dict["obs"]
                obs = obs.reshape(
                    self.num_envs, self.num_agents, *obs.shape[1:])
                rewards = rewards.reshape(self.num_envs, self.num_agents, -1)
                weights = torch.tensor([1., -1., 0., 0.], device=rewards.device)
                rewards = torch.sum(
                    rewards * weights, axis=-1, keepdim=True)
                dones = dones.reshape(self.num_envs, self.num_agents)
                data = obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic

                # insert data into buffer
                self.insert(data)

                # record statistics
                env_dones = dones.all(-1)
                if env_dones.any():
                    episode_info: Dict = infos.get("episode", {})
                    for k, v in episode_info.items():
                        episode_infos[k].append(v[env_dones])
                    total_episodes += env_dones.sum()

                inf_step_time += _inf_end - _step_start
                env_step_time += _env_end - _inf_end

            # compute return and update network
            self.compute()
            train_infos = self.train()

            total_env_steps += self.num_steps * self.num_envs

            # log information
            if iteration % self.log_interval == 0:
                end = time.perf_counter()
                print(
                    f"iteration: {iteration}/{self.max_iterations}, env steps: {total_env_steps}, episodes: {total_episodes}")
                print(
                    f"runtime: {env_step_time:.2f} (env), {inf_step_time:.2f} (inference), {time.perf_counter()-start:.2f} (total), fps: {total_env_steps/(end-start):.2f}")

                for k, v in episode_infos.items():
                    if v:
                        v = torch.cat(v).cpu().numpy().mean(
                            0)  # average over episodes
                        train_infos[k] = v
                    print(f"episode {k}: {v}")
                episode_infos.clear()

                self.log(train_infos, step=total_env_steps, tag="train")

            if total_env_steps > self.max_env_steps:
                break

    @torch.no_grad()
    def collect(self, step) -> Tuple[torch.Tensor, ...]:
        self.policy.prep_rollout()
        value, action, action_log_prob, rnn_states, rnn_states_critic \
            = self.policy.get_actions(
                self.buffer.share_obs[step].flatten(end_dim=1),
                self.buffer.obs[step].flatten(end_dim=1),
                self.buffer.rnn_states[step].flatten(end_dim=1),
                self.buffer.rnn_states_critic[step].flatten(end_dim=1),
                self.buffer.masks[step].flatten(end_dim=1))
        # [self.envs, agents, dim]
        values = value.reshape(self.n_rollout_threads, self.num_agents, -1)
        actions = action.reshape(self.n_rollout_threads, self.num_agents, -1)
        action_log_probs = action_log_prob.reshape(
            self.n_rollout_threads, self.num_agents, -1)
        rnn_states = rnn_states.reshape(
            self.n_rollout_threads, self.num_agents, *rnn_states.shape[1:])
        rnn_states_critic = rnn_states_critic.reshape(
            self.n_rollout_threads, self.num_agents, *rnn_states_critic.shape[1:])

        return values, actions, action_log_probs, rnn_states, rnn_states_critic

    def insert(self, data):
        obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        rnn_states[dones == True] = 0
        rnn_states_critic[dones == True] = 0

        dones_env = torch.all(dones, axis=1)

        masks = torch.ones((self.n_rollout_threads, self.num_agents, 1))
        masks[dones == True] = 0

        active_masks = torch.ones((self.n_rollout_threads, self.num_agents, 1))
        active_masks[dones == True] = 0
        active_masks[dones_env == True] = 1

        share_obs = obs

        self.buffer.insert(share_obs, obs, rnn_states, rnn_states_critic, actions,
                           action_log_probs, values, rewards, masks, active_masks=active_masks)

    def log(self, info: Dict[str, Any], step: int, tag: str = ""):
        wandb.log({f"{tag}/{k}": v for k, v in info.items()}, step=step)
