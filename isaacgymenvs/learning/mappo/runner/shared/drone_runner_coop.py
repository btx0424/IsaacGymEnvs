from collections import defaultdict
from dataclasses import dataclass
import logging
import time
from typing import Any, Dict, Tuple, Union, List
from isaacgymenvs.tasks.base.vec_task import MultiAgentVecTask
from isaacgymenvs.tasks.quadrotor import QuadrotorBase
from isaacgymenvs.tasks.quadrotor.base import TensorDict
import numpy as np
import torch
import wandb
import os
import random
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
    eval_interval: int = -1
    eval_episodes: int = 1024
    save_interval: int = 10

    rl_device: str = "cuda"

    use_cl: bool = False

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
        self.eval_interval = cfg.eval_interval
        self.eval_episodes = cfg.eval_episodes

        self.use_cl = cfg.use_cl

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
        self.env_steps_this_run = 0
        self.total_episodes = 0
        self.episodes_this_run = 0

        if self.use_cl:
            self.task_difficulty_collate_fn = lambda episode_infos, env_done: episode_infos["reward_capture"][env_done].mean(-1)

    def run(self):
        self.training = True
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


        episode_infos = defaultdict(lambda: [])
        metric = "success"
        if "best_" + metric not in wandb.run.summary.keys():
            wandb.run.summary["best_" + metric] = 0.0

        # setup CL
        if self.use_cl:
            tasks = TaskDist(capacity=16384)
            def sample_task(envs, env_ids, env_states: torch.Tensor):
                if self.training and len(tasks) > 0:
                    p = np.random.rand()
                    if p < 0.3:
                        envs.set_tasks(env_ids, tasks.sample(len(env_ids), "easy"), env_states)
                    elif p < 0.6:
                        envs.set_tasks(env_ids, tasks.sample(len(env_ids), "hard"), env_states)
                    else:
                        pass # leave the envs to use new tasks

            self.envs.reset_callbacks.append(sample_task)

        start = time.perf_counter()
        _last_log = time.perf_counter()

        for iteration in range(self.max_iterations):

            for step in range(self.num_steps):
                # Sample actions
                _step_start = time.perf_counter()

                # tensors by agent
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

                    if self.use_cl:
                        metrics = self.task_difficulty_collate_fn(episode_info, env_dones)
                        task_config = infos["task_config"][env_dones]
                        tasks.add(task_config=task_config, metrics=metrics)

                self.inf_step_time += _inf_end - _step_start
                self.env_step_time += _env_end - _inf_end
            
            # compute return and update network
            self.compute()
            train_infos = self.train()

            self.total_env_steps += self.num_steps * self.num_envs
            self.env_steps_this_run += self.num_steps * self.num_envs

            # log information
            if iteration % self.log_interval == 0:
                fps = (self.num_steps * self.num_envs * self.log_interval) / (time.perf_counter() - _last_log)
                progress_str = f"iteration: {iteration}/{self.max_iterations}, env steps: {self.total_env_steps}, episodes: {self.total_episodes}"
                performance_str = f"runtime: {self.env_step_time:.2f} (env), {self.inf_step_time:.2f} (inference), {time.perf_counter()-start:.2f} (total), fps: {fps:.2f}"
                
                print(progress_str)
                print(performance_str)

                for k, v in episode_infos.items():
                    v = np.array(v).mean(0)
                    if isinstance(v, np.ndarray):
                        train_infos[f"train/{k}"] = wandb.Histogram(v)
                        train_infos[f"train/{k}_mean"] = v.mean()
                    else:
                        train_infos[f"train/{k}"] = v
                    print(f"train/{k}: {v}")
                episode_infos.clear()

                train_infos["env_step"] = self.total_env_steps
                train_infos["episode"] = self.total_episodes
                train_infos["iteration"] = iteration

                if self.use_cl:
                    logging.info(f"Num of tasks: {len(tasks)}")
                    if len(tasks) > 0:
                        train_infos["train/task_distribution"] = wandb.Histogram(tasks.metrics.cpu().numpy())
                    
                self.log(train_infos)

                if train_infos.get(f"train/{metric}", wandb.run.summary["best_" + metric]) > wandb.run.summary["best_" + metric]:
                    wandb.run.summary["best_" + metric] = train_infos.get(f"train/{metric}")
                    logging.info(f"Saving best model with {metric}: {wandb.run.summary['best_' + metric]}")
                    self.save()
                
                _last_log = time.perf_counter()

            if iteration % self.eval_interval == 0:
                self.training = False
                self.eval(self.eval_episodes)
                self.training = True

            if self.env_steps_this_run > self.max_env_steps:
                wandb.run.summary["env_step"] = self.total_env_steps
                wandb.run.summary["episode"] = self.total_episodes
                wandb.run.summary["iteration"] = iteration
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

    def eval(self, eval_episodes):
        if self.env_steps_this_run == 0:
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
        else:
            for role, buffer in self.buffers.items():
                assert buffer.step == 0, buffer.step
            obs_dict = {
                role: {"obs": buffer.obs[buffer.step], "state": buffer.share_obs[buffer.step]} 
                for role, buffer in self.buffers.items()}
            rnn_states_dict = {
                role: (buffer.rnn_states[buffer.step].flatten(end_dim=1), buffer.rnn_states_critic[buffer.step].flatten(end_dim=1)) 
                for role, buffer in self.buffers.items()
            }
            masks_dict = {
                role: buffer.masks[buffer.step]
                for role, buffer in self.buffers.items()
            }

        already_reset = torch.zeros_like(self.envs.progress_buf, dtype=bool)

        episode_infos = defaultdict(lambda: [])
        # metric = "success_mean"
        # if "best_" + metric not in wandb.run.summary.keys():
        #     wandb.run.summary["best_" + metric] = 0.0

        episode_count = 0
        
        while episode_count < eval_episodes:

            for setp in range(self.num_steps):
                # tensors by agent
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

                step_result_dict, infos = self.envs.agents_step(action_dict)
                env_dones: torch.Tensor = infos["env_dones"]

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
                valid_envs = env_dones & already_reset
                if valid_envs.any():
                    episode_info: Dict[str, torch.Tensor] = infos.get("episode", {})
                    for k, v in episode_info.items():
                        episode_infos[k].extend(v[valid_envs].tolist())
                    episode_count += valid_envs.sum()
                    # progress.update(valid_envs.sum().item())
                already_reset |= env_dones

            for buffer in self.buffers.values(): buffer.after_update()

        eval_infos = {"env_step": self.total_env_steps}
        for k, v in episode_infos.items():
            v = np.array(v).mean(0)
            if isinstance(v, np.ndarray):
                eval_infos[f"eval/{k}"] = wandb.Histogram(v)
                eval_infos[f"eval/{k}_mean"] = v.mean()
            else:
                eval_infos[f"eval/{k}"] = v
            print(f"eval/{k}: {v}")
            
        self.log(eval_infos)

class TaskDist:
    def __init__(self, capacity: int=1000) -> None:
        self.capacity = capacity
        self.task_config = None
        self.metrics = None
    
    def add(self, task_config: TensorDict, metrics: torch.Tensor) -> None:
        if self.task_config is None:
            self.task_config = task_config
        else:
            self.task_config = torch.cat((self.task_config, task_config))[-self.capacity:]
        if self.metrics is None:
            self.metrics = metrics[-self.capacity:]
        else:
            self.metrics = torch.cat([self.metrics, metrics])[-self.capacity:]
    
    def sample(self, n: int, mode: str="easy") -> TensorDict:
        if mode == "easy":
            weights = self.metrics + 1
        elif mode == "hard":
            weights = 500 - self.metrics + 1
        else: 
            raise ValueError(f"Unknown mode: {mode}")
        # weights = weights / weights.sum()
        sampled_indices = weights.multinomial(n, replacement=True)
        # sampled_indices = random.choices(np.arange(len(self)), weights=weights, k=n)
        return self.task_config[sampled_indices]
        
    def __len__(self) -> int:
        return 0 if self.metrics is None else len(self.metrics)
