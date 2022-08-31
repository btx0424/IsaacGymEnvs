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

from isaacgymenvs.learning.mappo.algorithms.rmappo_new import MAPPOPolicy
from isaacgymenvs.learning.mappo.utils.data import TensorDict, LazyRolloutBuffer, group_by_agent
import pandas as pd
import plotly.express as px
from sklearn import manifold

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
    

def collect_episode_infos_(infos: Dict[str, List], tag: str) -> Dict:
    results = {}
    for k, v in infos.items():
        v = np.array(v).mean(0) # average over episodes
        if v.size > 1:
            results[f"{tag}/{k}"] = wandb.Histogram(v)
            results[f"{tag}/{k}_mean"] = v.mean()
        else:
            results[f"{tag}/{k}"] = v.item()
        print(f"{tag}/{k}: {v}")
    infos.clear()
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

        # CL config
        self.use_cl = cfg.use_cl
        self.progress_speed = cfg.progress_speed
        self.progress_threshold = 2.0

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
        for agent_type in self.agents:
            self.policies[agent_type] = MAPPOPolicy(
                self.all_args,
                self.num_agents[agent_type],
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
        
        lazybuffer = LazyRolloutBuffer(size=self.num_steps)

        def joint_policy(tensordict: TensorDict):
            grouped, others = group_by_agent(tensordict)
            for agent_type, agent_input in grouped.items():
                policy = self.policies[agent_type]
                agent_output = policy(agent_input)
                tensordict.update({f"{k}@{agent_type}":v for k, v in agent_output.items()})
            return tensordict
        
        def step_callback(env: MultiAgentVecTask, tensordict: TensorDict, step):
            lazybuffer.insert(tensordict.drop("rpms"))
            
        step_kwargs = {}
        if self.use_cl:
            task_buffer = LazyRolloutBuffer(size=8192)
            def on_reset(envs, envs_done: torch.BoolTensor, **kwargs):
                env_ids = envs_done.nonzero().squeeze(-1)
                # sample tasks to reset with
                if len(task_buffer) > 0 and random.random() < 0.5:
                    sample_idx = torch.randint(len(task_buffer), size=(len(env_ids),))
                    return_dict = task_buffer[sample_idx]
                else:
                    return_dict = {}
                # add new tasks
                if len(env_ids) > 0:
                    tasks: TensorDict = envs.extras["task"][env_ids]
                    tasks_valid: torch.BoolTensor = (tasks["success"]>=0.3) & (tasks["success"]<0.7)
                    if tasks_valid.any():
                        task_buffer.insert(tasks[tasks_valid], step=tasks_valid.sum())
                    for k, v in envs.extras["episode"][env_ids].items():
                        episode_infos[k].extend(v.tolist())
                        self.total_episodes += len(env_ids)
                return return_dict
        else:
            def on_reset(envs, envs_done: torch.BoolTensor, **kwargs):
                env_ids = envs_done.nonzero().squeeze(-1)
                if len(env_ids) > 0:
                    for k, v in envs.extras["episode"][env_ids].items():
                        episode_infos[k].extend(v.tolist())
                        self.total_episodes += len(env_ids)
                return {}

        step_kwargs["on_reset"] = on_reset

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
            train_infos = self.train(lazybuffer)
            iter_end = time.perf_counter()

            self.env_steps_this_run += self.num_steps * self.num_envs
            self.total_env_steps += self.num_steps * self.num_envs

            if iteration % self.log_interval == 0:
                fps = (self.num_steps * self.num_envs) / (iter_end - iter_start)
                logging.info(f"Iteration {iteration}/{self.max_iterations}, Env steps: {self.env_steps_this_run}/{self.max_env_steps} (this run), {self.total_env_steps} (total), Episodes: {self.total_episodes}")
                logging.info(f"FPS: {fps:.0f}, rollout: {rollout_end-iter_start:.2f}, train: {iter_end-rollout_end:.2f}")
                
                train_infos.update(collect_episode_infos_(episode_infos, "train"))
                train_infos["fps"] = fps
                if self.use_cl and len(task_buffer) > 0:
                    df = pd.DataFrame(task_buffer.select("success", "target_speeds").cpu().numpy())
                    train_infos["task_buffer"] = px.scatter(df, x="target_speeds", y="success")
                
                self.log(train_infos)

            if self.eval_interval > 0 and iteration % self.eval_interval == 0:
                eval_info = self.eval(self.eval_episodes, log=True, verbose=False)
                if self.progress_speed is not None and eval_info["eval/success@predator"] > self.progress_threshold:
                    start, step, end = self.progress_speed
                    self.envs.target_speed_dist = torch.distributions.Uniform(self.envs.target_speed_dist.low, min(self.envs.target_speed_dist.high+step, end))
                    logging.info(f"Increase target_speeds to be in range [{self.envs.target_speed_dist.low}, {self.envs.target_speed_dist.high}]")
                tensordict = self.envs.reset()

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
                break

        wandb.run.summary["env_step"] = self.total_env_steps
        wandb.run.summary["episode"] = self.total_episodes
        wandb.run.summary["iteration"] = iteration
        self.save(tag="end")
        return wandb.run.summary

    def train(self, buffer: TensorDict) -> Dict[str, Any]:
        train_infos = {}
        grouped, others= group_by_agent(buffer)
        for agent_type, policy in self.policies.items():
            policy.prep_training()
            agent_batch = grouped[agent_type]
            # reward shaping TODO: do it somewhere else
            agent_batch["rewards"] = agent_batch["rewards"].sum(-1, keepdim=True)
            agent_train_info = policy.train_on_batch(TensorDict(**agent_batch, **others))
            train_infos.update({f"{agent_type}.{k}": v for k, v in agent_train_info.items()})
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

    @torch.no_grad()
    def eval(self, eval_episodes, log=True, verbose=False):
        stamp_str = f"Eval at {self.total_env_steps}(total)/{self.env_steps_this_run}(this run) steps."
        logging.info(stamp_str)
        for agent_type, policy in self.policies.items():
            policy.prep_rollout()

        def eval_policy(tensordict: TensorDict):
            grouped, others = group_by_agent(tensordict)
            for agent_type, agent_input in grouped.items():
                policy = self.policies[agent_type]
                agent_output = policy.policy_op(agent_input, deterministic=True)
                tensordict.update({f"{k}@{agent_type}":v for k, v in agent_output.items()})
            return tensordict

        step_kwargs = {}
        episode_infos = defaultdict(list) # for scalars
        task_buffer = LazyRolloutBuffer(size=eval_episodes) # for tensors
        def on_reset(envs, envs_done: torch.BoolTensor, **kwargs):
            env_ids = envs_done.nonzero().squeeze(-1)
            if len(env_ids) > 0:
                for k, v in envs.extras["episode"][env_ids].items():
                    episode_infos[k].extend(v.tolist())
                tasks: TensorDict = envs.extras["task"][env_ids]
                task_buffer.insert(tasks, step=len(env_ids))
                assert (envs.extras["episode"][env_ids]["success@predator"] == tasks["success"]).all()
            return {}
        step_kwargs["on_reset"] = on_reset

        self.envs.eval()
        self.envs.rollout(max_episodes=eval_episodes, policy=eval_policy, step_kwargs=step_kwargs, verbose=verbose)
        eval_infos = {"env_step": self.total_env_steps}
        eval_infos.update(collect_episode_infos_(episode_infos, "eval"))

        if len(task_buffer) > 0:
            value_output = self.policies["predator"].value_op(task_buffer.rename(init_obs="obs"), disagreement=True).mean(1) # average over agents
            critic_features = self.policies["predator"].critics[0].get_feature(task_buffer["init_obs"][:, 0]) # first critic, first agent ...

            df = pd.DataFrame(task_buffer.select("success", "target_speeds", "collision").update(value_output).flatten().cpu().numpy())
            tsne = manifold.TSNE()
            embeddings = tsne.fit_transform(critic_features.cpu().numpy()) # [N, 2]
            x, y = embeddings.T
            eval_infos.update({
                "success": px.scatter(x=x, y=y, color=df["success"]),
                "target_speeds": px.scatter(x=x, y=y, color=df["target_speeds"]),
                "value_stds": px.scatter(x=x, y=y, color=df["value_stds"]),
                "success vs. target_speeds": px.scatter(df, x="target_speeds", y="success", size="value_stds", marginal_y="histogram", color="collision"),
            })

        task_buffer.clear()
        if log:
            self.log(eval_infos)
        return eval_infos
