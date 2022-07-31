from collections import defaultdict
from dataclasses import dataclass
import logging
import time
from typing import Any, Dict, Tuple, Union
from isaacgymenvs.tasks.base.vec_task import MultiAgentVecTask
from isaacgymenvs.tasks.quadrotor import QuadrotorBase
from isaacgymenvs.tasks.quadrotor.base import TensorDict
import numpy as np
import torch
import wandb
import os
from .base_runner import Runner

from isaacgymenvs.learning.mappo.algorithms.rmappo import MAPPOPolicy
from isaacgymenvs.learning.mappo.utils.shared_buffer import SharedReplayBuffer

Agent = str

@dataclass
class RunnerConfig:
    num_envs: int
    num_agents: int = 1

    max_iterations: int = 500
    max_env_steps: int = int(5e6)

    log_interval: int = 5
    save_interval: int = 10

    rl_device: str = "cuda"
@dataclass
class PPOConfig:
    num_steps: int = 16
    num_mini_batch: int = 8


class DroneRunner(Runner):
    def __init__(self, config):
        self.config = config

        cfg: RunnerConfig = config["cfg"]
        self.max_iterations = cfg.max_iterations
        self.max_env_steps = cfg.max_env_steps
        self.log_interval = cfg.log_interval

        self.num_envs = cfg.num_envs

        self.device = cfg.rl_device
        all_args = config["all_args"]

        if all_args.use_attn:
            envs: QuadrotorBase = config["envs"]
            obs_split = envs.obs_split
            envs.state_space = envs.obs_space = \
                [sum(num*dim for num, dim in obs_split), *obs_split]

        super().__init__(config)

        self.agents = self.envs.agents
        if len(self.agents) > 1:
            self.num_agents = {
                agent: getattr(self.envs, f"num_{agent}s") for agent in self.agents
            }
        else:
            self.num_agents = {self.agents[0]: self.envs.num_agents}
        self.num_steps = all_args.num_steps

        envs: MultiAgentVecTask = config["envs"]

        self.policies: Dict[str, MAPPOPolicy] = {}
        self.buffers: Dict[str, SharedReplayBuffer] = {}
        for agent in self.agents:
            self.policies[agent] = MAPPOPolicy(
                self.all_args,
                envs.obs_space,
                envs.state_space,
                envs.act_space)
            self.buffers[agent] = SharedReplayBuffer(
                self.all_args,
                self.num_envs,
                self.num_agents[agent],
                envs.obs_space,
                envs.state_space,
                envs.act_space)

        # timers & counters
        self.env_step_time = 0
        self.inf_step_time = 0
        
        self.total_env_steps = 0
        self.total_episodes = 0

    def run(self):
        obs_dict = self.envs.reset()

        rnn_states_dict = {
            agent: policy.get_initial_rnn_states(self.num_envs*self.num_agents[agent])
            for agent, policy in self.policies.items() if policy is not None
        }
        masks_dict = {
            agent: torch.ones((self.num_envs * self.num_agents[agent], 1), device=self.device)
            for agent in self.agents
        }
        for agent, agent_obs_dict in obs_dict.items():
            buffer: Union[SharedReplayBuffer, None] = self.buffers.get(agent)
            if buffer:
                buffer.obs[0] = agent_obs_dict["obs"].reshape(self.num_envs, self.num_agents[agent], -1)
                buffer.share_obs[0] = agent_obs_dict["state"].reshape(self.num_envs, self.num_agents[agent], -1)

        start = time.perf_counter()

        episode_infos = defaultdict(lambda: [])
        metric = "Episode/success/mean"
        metric_best = 0

        for iteration in range(self.max_iterations):

            if self.use_linear_lr_decay:
                self.policy.lr_decay(iteration, self.max_iterations)

            for step in range(self.num_steps):
                # Sample actions
                _step_start = time.perf_counter()

                action_dict = TensorDict()
                action_log_prob_dict = TensorDict()
                value_dict = TensorDict()
                rnn_state_actor_dict = TensorDict()
                rnn_state_critic_dict = TensorDict()

                for agent, agent_obs_dict in obs_dict.items():
                    policy: MAPPOPolicy = self.policies[agent]
                    policy.prep_rollout()
                    rnn_state_actor, rnn_state_critic = rnn_states_dict[agent]                 
                    masks = masks_dict[agent]
                    with torch.no_grad():
                        result_dict = policy.get_action_and_value(
                            share_obs=agent_obs_dict["state"],
                            obs=agent_obs_dict["obs"],
                            rnn_states_actor=rnn_state_actor,
                            rnn_states_critic=rnn_state_critic,
                            masks=masks
                        )
                    action_dict[agent] = result_dict["action"]
                    action_log_prob_dict[agent] = result_dict["action_log_prob"]
                    value_dict[agent] = result_dict["value"]
                    rnn_state_actor_dict[agent] = result_dict["rnn_state_actor"]
                    rnn_state_critic_dict[agent] = result_dict["rnn_state_critic"]

                _inf_end = time.perf_counter()
                step_result_dict, infos = self.envs.agents_step(action_dict)
                env_dones: torch.Tensor = infos["env_dones"]
                _env_end = time.perf_counter()

                for agent, (agent_obs_dict, agent_reward, agent_done) in step_result_dict.items():
                    # TODO: complete reward shaping
                    agent_reward = agent_reward.sum(-1)
                    
                    obs_dict[agent] = agent_obs_dict
                    masks_dict[agent] = (1.0 - agent_done).reshape(self.num_envs * self.num_agents[agent], 1)

                    buffer = self.buffers.get(agent)
                    if buffer:
                        num_agents = self.num_agents[agent]
                        data = (
                            agent_obs_dict["obs"].reshape(self.num_envs, num_agents, -1),
                            agent_reward.reshape(self.num_envs, num_agents, 1),
                            agent_done.reshape(self.num_envs, num_agents, 1),
                            value_dict[agent].reshape(self.num_envs, num_agents, 1),
                            action_dict[agent].reshape(self.num_envs, num_agents, -1),
                            action_log_prob_dict[agent].reshape(self.num_envs, num_agents, -1),
                            rnn_state_actor_dict[agent],
                            rnn_state_critic_dict[agent]
                        )
                        self.insert(data, buffer)

                # record statistics
                if env_dones.any():
                    episode_info: Dict[str, torch.Tensor] = infos.get("episode", {})
                    for k, v in episode_info.items():
                        episode_infos[k].extend(v[env_dones].tolist())
                    self.total_episodes += env_dones.sum()

                self.inf_step_time += _inf_end - _step_start
                self.env_step_time += _env_end - _inf_end
            
            # compute return and update network
            self.compute()
            train_infos = self.train()

            self.total_env_steps += self.num_steps * self.num_envs

            # log information
            if iteration % self.log_interval == 0:
                end = time.perf_counter()
                print(
                    f"iteration: {iteration}/{self.max_iterations}, env steps: {self.total_env_steps}, episodes: {self.total_episodes}")
                print(
                    f"runtime: {self.env_step_time:.2f} (env), {self.inf_step_time:.2f} (inference), {time.perf_counter()-start:.2f} (total), fps: {self.total_env_steps/(end-start):.2f}")

                for k, v in episode_infos.items():
                    v = torch.cat(v).cpu().numpy().mean(0)
                    train_infos[f"Episode/{k}"] = wandb.Histogram(v)
                    train_infos[f"Episode/{k}/mean"] = v.mean()
                    print(f"Episode/{k}: {v}")

                episode_infos.clear()

                train_infos["env_step"] = self.total_env_steps
                train_infos["episode"] = self.total_episodes
                train_infos["iteration"] = iteration
                self.log(train_infos)

                if train_infos.get(metric, metric_best) > metric_best:
                    metric_best = train_infos[metric]
                    logging.info(f"Saving best model with {metric}: {metric_best}")
                    self.save()

            if self.total_env_steps > self.max_env_steps:
                break

    def insert(self, data, buffer: SharedReplayBuffer = None):
        obs, rewards, dones, values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        if rnn_states is not None:
            rnn_states = rnn_states.reshape(self.num_envs, obs.size(1), *rnn_states.shape[1:])
            rnn_states[dones] = 0
        if rnn_states_critic is not None:
            rnn_states_critic = rnn_states_critic.reshape(self.num_envs, obs.size(1), *rnn_states_critic.shape[1:])
            rnn_states_critic[dones] = 0

        masks = torch.ones_like(dones)
        masks[dones] = 0

        share_obs = obs

        buffer.insert(
            share_obs, obs, 
            rnn_states, rnn_states_critic, 
            actions, action_log_probs, 
            values, rewards, masks)

    @torch.no_grad()
    def compute(self):
        for agent, policy in self.policies.items():
            buffer = self.buffers.get(agent)
            next_values = policy.get_values(
                buffer.share_obs[-1].flatten(end_dim=1),
                buffer.rnn_states_critic[-1].flatten(end_dim=1),
                buffer.masks[-1].flatten(end_dim=1))
            next_values = next_values.reshape(self.num_envs, self.num_agents[agent], 1)
            buffer.compute_returns(next_values, policy.value_normalizer)
    
    def train(self) -> Dict[str, Any]:
        train_infos = {}
        for agent, policy in self.policies.items():
            if isinstance(policy, MAPPOPolicy):
                buffer = self.buffers[agent]
                policy.prep_training()
                train_infos[agent] = policy.train(buffer)      
                buffer.after_update()
        self.log_system()
        return train_infos

    def log(self, info: Dict[str, Any]):
        wandb.log({f"{k}": v for k, v in info.items()})

    def save(self):
        logging.info(f"Saving models to {wandb.run.dir}")
        checkpoint = {}
        for agent, policy in self.policies.items():
            checkpoint[agent] = {
                "actor": policy.actor.state_dict(),
                "critic": policy.critic.state_dict(),
            }
        checkpoint["episodes"] = self.total_episodes
        checkpoint["env_steps"] = self.total_env_steps
        torch.save(checkpoint, os.path.join(wandb.run.dir, "checkpoint.pt"))

    def restore(self):
        logging.info(f"Restoring models from {wandb.run.dir}")
        checkpoint = torch.load(os.path.join(wandb.run.dir, "checkpoint.pt"))
        for agent, policy in self.policies.items():
            policy.actor.load_state_dict(checkpoint[agent]["actor"])
            policy.critic.load_state_dict(checkpoint[agent]["critic"])
        self.total_episodes = checkpoint["episodes"]
        self.total_env_steps = checkpoint["env_steps"]
        