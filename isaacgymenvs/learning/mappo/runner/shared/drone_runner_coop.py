from collections import defaultdict
from dataclasses import dataclass
import logging
import time
from typing import Any, Callable, Dict, Mapping, Optional, Tuple, Union, List
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
    
from isaacgymenvs.learning.mappo.utils.data import TensorDict
class LazyRolloutBuffer(TensorDict):
    def __init__(self, size=64, stack_dim=0):
        super().__init__()
        self.size = size
        self.stack_dim = stack_dim
        self._step = 0

    def insert(self, dict: Dict[str, torch.Tensor], create_next=False, step=True):
        for k, v in dict.items():
            if k not in self.keys():
                if create_next:
                    size = (*v.shape[:self.stack_dim], self.size+1, *v.shape[self.stack_dim:])
                    tensor = self[f"_{k}"] = torch.zeros(size, dtype=v.dtype, device=v.device)
                    self[k], self[f"next_{k}"] = tensor[:-1], tensor[1:]
                else:
                    size = (*v.shape[:self.stack_dim], self.size, *v.shape[self.stack_dim:])
                    self[k] = torch.zeros(size, dtype=v.dtype, device=v.device)
            self[k][self._step] = v
        if step:
            self._step = (self._step + 1) % self.size
        
    def update(self, dict):
        raise NotImplementedError

def group_by_agent(tensordict: Dict[str, Any]):
    grouped = defaultdict(TensorDict)
    others = TensorDict()
    for k, v in tensordict.items():
        k = k.split("@")
        if len(k) > 1:
            k, agent = k
            grouped[agent][k] = v
        else:
            others[k[0]] = v
    return grouped, others

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
        self.save_interval = 200
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

    def run(self):
        self.training = True
        tensordict = self.envs.reset()
        episode_infos = defaultdict(list)

        metric_key = "train/success@predator"
        if "best_" + metric_key not in wandb.run.summary.keys(): 
            wandb.run.summary["best_" + metric_key] = 0
        
        def init_buffer(tensordict):
            for agent_type, buffer in self.buffers.items():
                buffer.obs[0] = tensordict[f"obs@{agent_type}"]
                buffer.share_obs[0] = tensordict[f"obs@{agent_type}"]
        
        init_buffer(tensordict)
        
        lazybuffer = LazyRolloutBuffer(size=self.num_steps)

        def joint_policy(tensordict: TensorDict):
            for agent_type, policy in self.policies.items():
                result_dict = policy.get_action_and_value(
                    share_obs=tensordict[f"obs@{agent_type}"].flatten(0, 1),
                    obs=tensordict[f"obs@{agent_type}"].flatten(0, 1),
                )
                tensordict.update({
                    f"{k}@{agent_type}":v.reshape(self.num_envs, self.num_agents[agent_type], -1) 
                    for k, v in result_dict.items() if v is not None})
            # grouped, others = group_by_agent(tensordict)
            # for agent_type, agent_input in grouped.items():
            #     policy = self.policies[agent_type]
            #     agent_output = policy(agent_input)
            #     tensordict.update({f"{k}@{agent_type}":v for k, v in agent_output.items()})
            return tensordict
        
        def step_callback(env, tensordict: TensorDict, step):
            
            for agent_type, buffer in self.buffers.items():
                buffer.insert(**TensorDict(dict(
                    share_obs=tensordict[f"next_obs@{agent_type}"],
                    obs=tensordict[f"next_obs@{agent_type}"],
                    actions=tensordict[f"actions@{agent_type}"],
                    action_log_probs=tensordict[f"action_log_probs@{agent_type}"],
                    rewards=tensordict[f"rewards@{agent_type}"].sum(-1, keepdim=True),
                    masks=1.0 - tensordict[f"dones@{agent_type}"],
                    value_preds=tensordict[f"values@{agent_type}"],
                )))

            env_done = tensordict["env_done"].squeeze(-1)
            if env_done.any():
                for k, v in env.extras["episode"][env_done].items():
                    episode_infos[k].extend(v.tolist())
                self.total_episodes += env_done.sum().item()
        
            # lazybuffer.insert(tensordict.drop("rpms", "env_done"))
            
        step_kwargs = {}
        if self.use_cl:
            class TensorListBuffer:
                def __init__(self, capacity:int=8192):
                    self._dict = defaultdict(list)
                    self.capacity = capacity
                
                def update(self, **kwargs):
                    for k, v in kwargs.items():
                        self._dict[k].extend(v)
                        self._dict[k] = self._dict[k][-self.capacity:]
                        self._len = (self._len + len(v)) % self.capacity
                
                def sample(self, n, keys: List[str]=None) -> Dict[str, torch.Tensor]:
                    sample_idx = torch.randint(n)
                    if keys is not None:
                        return {k: torch.stack(v)[sample_idx] for k, v in self._dict if k in keys}
                    else:
                        return {k: torch.stack(v)[sample_idx] for k, v in self._dict}

                def __len__(self):
                    return self._len

            self.tasks = TensorListBuffer()
            step_kwargs["task_buffer"] = self.tasks
            step_kwargs["sample_task_p"] = 0.5

        self.envs.train()
        for iteration in range(self.max_iterations):
            iter_start = time.perf_counter()
            for agent_type, policy in self.policies.items():
                policy.prep_rollout()
            with torch.no_grad():
                tensordict = self.envs.rollout(
                    tensordict=tensordict,
                    max_steps=self.num_steps, 
                    policy=joint_policy, 
                    step_kwargs=step_kwargs,
                    callback=step_callback)
            rollout_end = time.perf_counter()
            train_infos = self.train()
            # train_infos = {}
            # grouped, others= group_by_agent(lazybuffer)
            # for agent_type, policy in self.policies.items():
            #     agent_batch = grouped[agent_type]
            #     # reward shaping TODO: do it somewhere else
            #     agent_batch["rewards"] = agent_batch["rewards"].sum(-1, keepdim=True)
            #     agent_train_info = policy.train_on_batch(TensorDict(**agent_batch, **others))
            #     train_infos.update({f"{agent_type}/{k}": v for k, v in agent_train_info.items()})

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
                self.eval(self.eval_episodes, log=True, verbose=False)
                tensordict = self.envs.reset()
                init_buffer(tensordict)

            if (
                    metric_key in train_infos.keys() and
                    train_infos.get(metric_key) > wandb.run.summary[f"best_{metric_key}"]
                ):
                wandb.run.summary[f"best_{metric_key}"] = train_infos.get(metric_key)
                logging.info(f"Saving best model with {metric_key}: {wandb.run.summary[f'best_{metric_key}']}")
                self.save(tag="best", info=train_infos)
            
            if self.save_interval > 0 and iteration % self.save_interval == 0:
                self.save(info=train_infos)

            if self.env_steps_this_run > self.max_env_steps:
                wandb.run.summary["env_step"] = self.total_env_steps
                wandb.run.summary["episode"] = self.total_episodes
                wandb.run.summary["iteration"] = iteration
                self.save(tag="end")
                break

        return wandb.run.summary

    def train(self) -> Dict[str, Any]:
        train_infos = {}
        for agent_type, policy in self.policies.items():
            buffer = self.buffers[agent_type]
            with torch.no_grad():
                next_values = policy.get_value(
                    buffer.share_obs[-1].flatten(end_dim=1),
                    buffer.rnn_states_critic[-1],
                    buffer.masks[-1].flatten(end_dim=1))
                next_values = next_values.reshape(self.num_envs, self.num_agents[agent_type], 1)
                buffer.compute_returns(next_values, policy.value_normalizer)
            policy.prep_training()
            train_infos[agent_type] = policy.train(buffer)      
            buffer.after_update()
        self.log_system()
        return train_infos

    def log(self, info: Dict[str, Any]):
        info["env_step"] = self.total_env_steps
        wandb.log(info)

    def save(self, tag: Optional[str]=None, info: Optional[Dict[str, Any]]=None):
        logging.info(f"Saving models to {wandb.run.dir}")
        checkpoint = {}
        for agent_type, policy in self.policies.items():
            checkpoint[agent_type] = policy.state_dict()
        checkpoint["episodes"] = self.total_episodes
        checkpoint["env_steps"] = self.total_env_steps

        if tag is not None:
            checkpoint_name = f"checkpoint_{tag}.pt"
        else:
            checkpoint_name = f"checkpoint_{self.total_env_steps}"

        torch.save(checkpoint, os.path.join(wandb.run.dir, checkpoint_name))

    def restore(self, reset_steps=False, tag: Optional[str]=None):
        logging.info(f"Restoring models from {wandb.run.dir}")
        if tag is not None:
            checkpoint_name = f"checkpoint_{tag}.pt"
        else:
            checkpoint_name = "checkpoint.pt"

        checkpoint = torch.load(os.path.join(wandb.run.dir, checkpoint_name), map_location=self.device)
        for agent_type, policy in self.policies.items():
            policy.load_state_dict(checkpoint[agent_type])
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

        self.envs.eval()
        with torch.no_grad():
            self.envs.rollout(max_episodes=eval_episodes, policy=eval_policy, callback=step_callback, verbose=verbose)

        eval_infos = {"env_step": self.total_env_steps}
        eval_infos.update(collect_episode_infos(episode_infos, "eval"))
        if "target_speed" in episode_infos.keys():
            import plotly.express as px
            import pandas as pd
            from sklearn import manifold

            df = pd.DataFrame({
                "target_speed": np.array(episode_infos["target_speed"]).flatten(),
                "success": np.array(episode_infos["success@predator"]).flatten(),
                "reward_capture": np.array(episode_infos["reward_capture@predator"]).flatten(),
                })
            feature, value = self.policies["predator"].critics[0].get_feature_and_value()
            manifold.TSNE()
            wandb.log({
                "success vs. target_speed": px.density_heatmap(df, x="target_speed", y="success", marginal_y="histogram"),
                "capture_reward vs. target_speed": px.density_heatmap(df, x="target_speed", y="success", marginal_y="histogram"),
                # "feature t-SNE plot": None
            })
        if log:
            self.log(eval_infos)
        return eval_infos

def update_recursive_(a: Dict, b: Dict):
    for k, v in b.items():
        if isinstance(a.get(k), Dict) and isinstance(v, Dict):
            update_recursive_(a[k], v)
        else:
            a[k] = v
