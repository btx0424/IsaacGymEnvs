from collections import defaultdict
from dataclasses import dataclass
import logging
import time
from typing import Any, Callable, Dict, Tuple, Union, List
from isaacgymenvs.tasks.base.vec_task import MultiAgentVecTask
from isaacgymenvs.tasks.quadrotor import QuadrotorBase
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

class TensorDict(Dict[str, torch.Tensor]):
    def reshape(self, *shape: int):
        return TensorDict({key: value.reshape(*shape) for key, value in self.items()})

    def flatten(self, start_dim: int = 0, end_dim: int = -1):
        return TensorDict({key: value.flatten(start_dim, end_dim) for key, value in self.items()})

    def group(self, get_group: Callable[[str], Union[str, None]]):
        groups = defaultdict(TensorDict)
        others = TensorDict()
        for k, v in self.items():
            group = get_group(k)
            if group: groups[group][k] = v
            else: others[k] = v
        return TensorDict(**groups), others

    def group_by_agent(self):
        return self.group(self.get_agent_group)
    
    @staticmethod
    def get_agent_group(k: str):
        idx = k.find('@')
        return k[:idx] if idx != -1 else None

def collect_episode_infos(infos: Dict[str, List], tag: str) -> Dict:
    results = {}
    for k, v in infos.items():
        v = np.array(v).mean(0) # average over episodes
        if v.size > 1:
            results[f"{tag}/{k}"] = wandb.Histogram(v)
            results[f"{tag}/{k}_mean"] = v.mean()
        else:
            results[f"{tag}/{k}"] = v.item()
        print(f"{tag}/{k}: {v}")
    return results

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

        super().__init__(config)

        self.agents = self.envs.agent_types
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
                envs.obs_space,
                envs.act_space)
            self.buffers[agent] = SharedReplayBuffer(
                self.all_args,
                self.num_envs,
                self.num_agents[agent],
                envs.obs_space,
                envs.obs_space,
                envs.act_space)

        # timers & counters
        self.env_step_time = 0
        self.inf_step_time = 0
        self.train_time = 0

        self.total_env_steps = 0
        self.env_steps_this_run = 0
        self.total_episodes = 0
        self.episodes_this_run = 0

        if self.use_cl:
            self.task_difficulty_collate_fn = lambda episode_infos: episode_infos["reward_capture"].mean(-1)

    def run(self):
        self.training = True
        tensordict = self.envs.reset()
        episode_infos = defaultdict(list)

        metric_key = "train/reward_capture@predator_mean"
        if "best_" + metric_key not in wandb.run.summary.keys(): 
            wandb.run.summary["best_" + metric_key] = 0
        
        def init_buffer(tensordict):
            for agent_type, buffer in self.buffers.items():
                buffer.obs[0] = tensordict[f"obs@{agent_type}"]
                buffer.share_obs[0] = tensordict[f"obs@{agent_type}"]
        
        init_buffer(tensordict)

        def joint_policy(tensordict: TensorDict):
            for agent_type, policy in self.policies.items():
                result_dict = policy.get_action_and_value(
                    share_obs=tensordict[f"obs@{agent_type}"].flatten(0, 1),
                    obs=tensordict[f"obs@{agent_type}"].flatten(0, 1),
                )
                tensordict[f"actions@{agent_type}"] = result_dict["action"].reshape(self.num_envs, self.num_agents[agent_type], -1)
                tensordict[f"value@{agent_type}"] = result_dict["value"].reshape(self.num_envs, self.num_agents[agent_type], -1)
                tensordict[f"action_log_prob@{agent_type}"] = result_dict["action_log_prob"].reshape(self.num_envs, self.num_agents[agent_type], -1)
            return tensordict
        

        def step_callback(env, tensordict, step):
            for agent_type, buffer in self.buffers.items():
                buffer.insert(**TensorDict(dict(
                    share_obs=tensordict[f"next_obs@{agent_type}"],
                    obs=tensordict[f"next_obs@{agent_type}"],
                    actions=tensordict[f"actions@{agent_type}"],
                    action_log_probs=tensordict[f"action_log_prob@{agent_type}"],
                    rewards=tensordict[f"reward@{agent_type}"].sum(-1),
                    masks=1.0 - tensordict[f"done@{agent_type}"],
                    value_preds=tensordict[f"value@{agent_type}"],
                )).reshape(self.num_envs, self.num_agents[agent_type], -1))

            env_done = tensordict["env_done"].squeeze(-1)
            if env_done.any():
                for k, v in env.extras["episode"][env_done].items():
                    episode_infos[k].extend(v.tolist())
                self.total_episodes += env_done.sum().item()

        for iteration in range(self.max_iterations):
            iter_start = time.perf_counter()
            for agent_type, policy in self.policies.items():
                policy.prep_rollout()
            with torch.no_grad():
                tensordict = self.envs.rollout(
                    tensordict=tensordict,
                    max_steps=self.num_steps, 
                    policy=joint_policy, 
                    callback=step_callback)
            rollout_end = time.perf_counter()
            self.compute()
            train_infos = self.train()
            iter_end = time.perf_counter()

            self.env_steps_this_run += self.num_steps * self.num_envs
            self.total_env_steps += self.num_steps * self.num_envs

            if iteration % self.log_interval == 0:
                fps = (self.num_steps * self.num_envs) / (iter_end - iter_start)
                logging.info(f"Iteration {iteration}/{self.max_iterations}, Env steps: {self.env_steps_this_run}/{self.max_env_steps} (this run), {self.total_env_steps} (total), Episodes: {self.total_episodes}")
                logging.info(f"FPS: {fps:.0f}, rollout: {rollout_end-iter_start:.2f}, train: {iter_end-rollout_end:.2f}")
                
                train_infos.update(collect_episode_infos(episode_infos, "train"))
                train_infos["fps"] = fps
                self.log(train_infos)

                episode_infos.clear()

            if self.eval_interval > 0 and iteration % self.eval_interval == 0:
                self.eval(self.eval_episodes, log=True, pbar=False)
                tensordict = self.envs.reset()
                init_buffer(tensordict)

            if (
                    metric_key in train_infos.keys() and
                    train_infos.get(metric_key) > wandb.run.summary[f"best_{metric_key}"]
                ):
                wandb.run.summary[f"best_{metric_key}"] = train_infos.get(metric_key)
                logging.info(f"Saving best model with {metric_key}: {wandb.run.summary[f'best_{metric_key}']}")
                self.save()

            if self.env_steps_this_run > self.max_env_steps:
                wandb.run.summary["env_step"] = self.total_env_steps
                wandb.run.summary["episode"] = self.total_episodes
                wandb.run.summary["iteration"] = iteration
                break

        return wandb.run.summary

        distributions = defaultdict(list)

        # setup CL
        if self.use_cl:
            tasks = TaskDist(capacity=16384)
            def sample_task(envs, env_ids, env_states: torch.Tensor):
                # z = (1-self.env_steps_this_run / self.max_env_steps)*0.7
                z = 0.3
                if self.training and len(tasks) > 0:
                    p = np.random.rand()
                    if p < z:
                        sampled_tasks = tasks.sample(len(env_ids), "hard")
                        if sampled_tasks is not None:
                            envs.set_tasks(env_ids, sampled_tasks, env_states)
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

                for agent, agent_obs_dict in obs_dict.items():
                    policy: MAPPOPolicy = self.policies[agent]
                    policy.prep_rollout()
                    rnn_state_actor, rnn_state_critic = rnn_states_dict[agent]                 
                    masks = masks_dict[agent]
                    with torch.no_grad():
                        assert not torch.isnan(agent_obs_dict["obs"]).any()
                        result_dict = policy.get_action_and_value(
                            share_obs=agent_obs_dict["state"].flatten(end_dim=1),
                            obs=agent_obs_dict["obs"].flatten(end_dim=1),
                            rnn_states_actor=rnn_state_actor,
                            rnn_states_critic=rnn_state_critic,
                            masks=masks
                        )
                    action_dict[agent] = result_dict["action"]
                    action_log_prob_dict[agent] = result_dict["action_log_prob"]
                    value_dict[agent] = result_dict["value"]
                    rnn_states_dict[agent] = (result_dict["rnn_state_actor"], result_dict["rnn_state_critic"])

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
                            rnn_states_dict[agent][0],
                            rnn_states_dict[agent][1]
                        )
                        self.insert(data, buffer)

                # record statistics
                if env_dones.any():
                    episode_info = infos["episode"][env_dones]
                    task_config: TensorDict = infos.get("task_config")
                    for k, v in episode_info.items():
                        episode_infos[k].extend(v.tolist())

                    if self.use_cl:
                        metrics = self.task_difficulty_collate_fn(episode_info)
                        tasks.add(task_config=task_config[env_dones], metrics=metrics)

                    if task_config is not None:
                        for k, v in task_config[env_dones].items():
                            if v.squeeze(-1).dim() == 1: # e.g., target_speed
                                distributions[k].extend(v.squeeze(-1).tolist())

                    self.total_episodes += env_dones.sum()

                self.inf_step_time += _inf_end - _step_start
                self.env_step_time += _env_end - _inf_end
            
            # compute return and update network
            self.compute()
            _train_start = time.perf_counter()
            train_infos = self.train()
            _train_end = time.perf_counter()
            self.train_time += _train_end - _train_start

            self.total_env_steps += self.num_steps * self.num_envs
            self.env_steps_this_run += self.num_steps * self.num_envs

            # log information
            if iteration % self.log_interval == 0:
                fps = (self.num_steps * self.num_envs * self.log_interval) / (time.perf_counter() - _last_log)
                progress_str = f"iteration: {iteration}/{self.max_iterations}, env steps: {self.total_env_steps}, episodes: {self.total_episodes}"
                performance_str = f"runtime: {self.env_step_time:.2f} (env), {self.inf_step_time:.2f} (inference), {self.train_time:.2f} (train), {time.perf_counter()-start:.2f} (total), fps: {fps:.2f}"
                
                logging.info(progress_str)
                logging.info(performance_str)

                train_infos.update(collect_episode_infos(episode_infos, "train"))
                episode_infos.clear()

                for k, v in distributions.items():
                    v = np.array(v).reshape(-1)
                    np_histogram = np.histogram(v, density=True)
                    train_infos[f"train/{k}_distribution"] = wandb.Histogram(np_histogram=np_histogram)
                distributions.clear()

                train_infos["env_step"] = self.total_env_steps
                train_infos["episode"] = self.total_episodes
                train_infos["iteration"] = iteration
                train_infos["fps"] = fps

                if self.use_cl:
                    logging.info(f"Num of tasks: {len(tasks)}")
                    if len(tasks) > 0:
                        train_infos["train/task_buffer_difficulty"] = wandb.Histogram(tasks.metrics.cpu().numpy())
                    
                self.log(train_infos)

                if train_infos.get(f"train/{metric}", wandb.run.summary["best_" + metric]) > wandb.run.summary["best_" + metric]:
                    wandb.run.summary["best_" + metric] = train_infos.get(f"train/{metric}")
                    logging.info(f"Saving best model with {metric}: {wandb.run.summary['best_' + metric]}")
                    self.save()
                
                _last_log = time.perf_counter()

            if self.eval_interval > 0 and iteration % self.eval_interval == 0:
                self.training = False
                self.eval(self.eval_episodes)
                for agent, policy in self.policies.items():
                    policy.actor.train()
                    policy.critic.train()
                self.training = True

            if self.env_steps_this_run > self.max_env_steps:
                wandb.run.summary["env_step"] = self.total_env_steps
                wandb.run.summary["episode"] = self.total_episodes
                wandb.run.summary["iteration"] = iteration
                break

    @torch.no_grad()
    def compute(self):
        for agent, policy in self.policies.items():
            buffer = self.buffers.get(agent)
            next_values = policy.get_values(
                buffer.share_obs[-1].flatten(end_dim=1),
                buffer.rnn_states_critic[-1],
                buffer.masks[-1].flatten(end_dim=1))
            next_values = next_values.reshape(self.num_envs, self.num_agents[agent], 1)
            buffer.compute_returns(next_values, policy.value_normalizer)
    
    def train(self) -> Dict[str, Any]:
        train_infos = {}
        for agent_type, policy in self.policies.items():
            buffer = self.buffers[agent_type]
            policy.prep_training()
            train_infos[agent_type] = policy.train(buffer)      
            buffer.after_update()
        self.log_system()
        return train_infos

    def log(self, info: Dict[str, Any]):
        info["env_step"] = self.total_env_steps
        wandb.log(info)

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

    def restore(self, reset_steps=False):
        logging.info(f"Restoring models from {wandb.run.dir}")
        checkpoint = torch.load(os.path.join(wandb.run.dir, "checkpoint.pt"), map_location=self.device)
        for agent, policy in self.policies.items():
            policy.actor.load_state_dict(checkpoint[agent]["actor"])
            policy.critic.load_state_dict(checkpoint[agent]["critic"])
        if not reset_steps:
            self.total_episodes = checkpoint["episodes"]
            self.total_env_steps = checkpoint["env_steps"]

    def eval(self, eval_episodes, log=True, verbose=False):
        stamp_str = f"Eval at {self.total_env_steps}(total)/{self.env_steps_this_run}(this run) steps."
        logging.info(stamp_str)
        for agent_type, policy in self.policies.items():
            policy.prep_rollout()

        episode_infos = defaultdict(list)
        def step_callback(env, tensordict, step):
            env_done = tensordict["env_done"].squeeze(-1)
            if env_done.any():
                for k, v in env.extras["episode"][env_done].items():
                    episode_infos[k].extend(v.tolist())
        
        def eval_policy(tensordict):
            for agent_type, policy in self.policies.items():
                actions, _ = policy.act(
                    obs=tensordict[f"obs@{agent_type}"],
                    rnn_states_actor=None #tensordict[f"rnn_states_actor@{agent_type}"],
                )
                tensordict[f"actions@{agent_type}"] = actions
            return tensordict

        with torch.no_grad():
            self.envs.rollout(max_episodes=eval_episodes, policy=eval_policy, callback=step_callback)

        eval_infos = {"env_step": self.total_env_steps}
        eval_infos.update(collect_episode_infos(episode_infos, "eval"))
        
        if log:
            self.log(eval_infos)
        return eval_infos

