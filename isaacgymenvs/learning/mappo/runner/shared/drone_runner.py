from argparse import Namespace
from collections import defaultdict
from dataclasses import dataclass
import logging
import time
from typing import Any, Dict, Tuple, Union, List
from isaacgymenvs.tasks.base.vec_task import MultiAgentVecTask
from isaacgymenvs.tasks.quadrotor import QuadrotorBase
import numpy as np
import torch
import torch.nn as nn
from torchrl.data.tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.modules.distributions.continuous import NormalParamWrapper
from torchrl.modules.models.models import MLP
import wandb
import os
import random
from .base_runner import Runner

from isaacgymenvs.learning.mappo.algorithms.rmappo import MAPPOPolicy
from isaacgymenvs.learning.mappo.utils.shared_buffer import SharedReplayBuffer
import torch.distributions as distributions

from torchrl.modules import (
    MLP,
    ActorCriticWrapper,
    NormalParamWrapper,
    TanhNormal,
    TruncatedNormal,
    ValueOperator,
)

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
    num_minibatches: int = 8
    ppo_epochs: int = 4

    actor_lr: float = 0.005
    critic_lr: float = 0.005

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

from torchrl.modules.tensordict_module.sequence import TensorDictSequence
from torchrl.modules.tensordict_module.common import TensorDictModule, TensorDictModuleWrapper
from torchrl.modules.tensordict_module.actors import ActorCriticWrapper, ProbabilisticActor
from torchrl.objectives.returns.advantages import GAE
from torchrl.objectives.costs.ppo import ClipPPOLoss
from torchrl.envs.utils import set_exploration_mode, step_tensordict
from torch.profiler import profile, record_function, ProfilerActivity


def generalized_advantage_estimate(
    gamma: float,
    lmbda: float,
    state_value: torch.Tensor, # (num_steps, batch_size)
    next_state_value: torch.Tensor, # (num_steps, batch_size)
    reward: torch.Tensor,
    done: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    not_done = 1 - done.to(next_state_value.dtype)
    time_steps = not_done.shape[0]
    advantage = torch.zeros_like(reward)
    prev_advantage = 0

    for t in reversed(range(time_steps)):
        delta = (
            reward[t]
            + (gamma * next_state_value[t] * not_done[t])
            - state_value[t]
        )

        prev_advantage = advantage[t] = delta + (
            gamma * lmbda * prev_advantage * not_done[t]
        )
        
    value_target = advantage + state_value

    return advantage, value_target

class Normal(distributions.Normal):

    def log_prob(self, value):
        return super().log_prob(value).sum(-1, keepdim=True)
    
    def entropy(self):
        return super().entropy().sum(-1, keepdim=True)

class FixedNormalParamWrapper(nn.Module):
    def __init__(self, module: nn.Linear):
        super().__init__()
        self.module = module
        self.log_std = nn.Parameter(torch.zeros(module.out_features))
    
    def forward(self, hidden):
        loc = self.module(hidden)
        return loc, torch.exp(self.log_std).expand_as(loc)
    
class ActorCriticPolicy(ActorCriticWrapper):
    def __init__(self, obs_spec, act_spec, cfg: PPOConfig):
        self.cfg = cfg
        obs_dim = obs_spec.shape[0]
        action_dim = act_spec.shape[0]

        module_action = []
        module_action.append(TensorDictModule(
            MLP(in_features=obs_dim, out_features=256, num_cells=[256, 256]), 
            in_keys=[f"obs"], out_keys=["hidden"]))
        # module_action.append(TensorDictModule(
        #     NormalParamWrapper(nn.Linear(256, action_dim*2)), 
        #     in_keys=["hidden"], out_keys=["loc", "scale"]))
        module_action.append(TensorDictModule(
            FixedNormalParamWrapper(nn.Linear(256, action_dim)), 
            in_keys=["hidden"], out_keys=["loc", "scale"]))
        module_action = TensorDictSequence(*module_action)

        policy_op = ProbabilisticActor(
            module=module_action,
            dist_param_keys=["loc", "scale"],
            out_key_sample=[f"actions"],
            # distribution_class=TanhNormal,
            distribution_class=Normal,
            return_log_prob=True,
        )

        module_value = []
        module_value.append(
            TensorDictModule(MLP(in_features=obs_dim, out_features=256, num_cells=[256, 256]), in_keys=["state"], out_keys=["hidden"]))
        module_value.append(
            TensorDictModule(nn.Linear(256, 1), in_keys=["hidden"], out_keys=["state_value"]))
        value_op = TensorDictSequence(*module_value)
        super().__init__(policy_op, value_op)

        self.policy_opt = torch.optim.Adam(policy_op.parameters(), lr=cfg.actor_lr)
        self.value_opt = torch.optim.Adam(value_op.parameters(), lr=cfg.critic_lr)

    def forward(self, tensordict: TensorDictBase, tensordict_out=None, **kwargs) -> TensorDictBase:
        result_td = super().forward(tensordict, tensordict_out, **kwargs)
        result_td.del_("hidden")
        result_td.del_("loc")
        result_td.del_("scale")
        return result_td

    def update(self, batch: TensorDictBase) -> Dict:
        batch["action"] = batch["actions"]
        batch["reward"] = batch["reward"].sum(-1, keepdim=True)
        with torch.no_grad():
            last_state = TensorDict({"state": batch["next_state"][-1]}, batch_size=[])
            last_state_value = self.get_value_operator()(last_state)["state_value"]
            next_state_value = torch.cat([batch["state_value"][1:], last_state_value.unsqueeze(0)], dim=0)
            batch["advantage"], batch["value_target"] = generalized_advantage_estimate(0.99, 0.95, batch["state_value"], next_state_value, batch["reward"], batch["done"])
        train_info = {}
        self.train()
        for ppo_epoch in range(self.cfg.ppo_epochs):
            for minibatch in self.make_dataset(batch, self.cfg.num_minibatches):
                dist, *_ = self.get_policy_operator().get_dist(minibatch)
                log_probs = dist.log_prob(minibatch["action"])
                ratio = torch.exp(log_probs - minibatch["sample_log_prob"])
                policy_loss = torch.clamp(ratio, 1 - self.cfg.clip_param, 1 + self.cfg.clip_param) * minibatch["advantage"]
                policy_loss = torch.min(policy_loss, ratio * minibatch["advantage"])
                self.policy_opt.zero_grad()
                policy_loss.mean().backward()
                self.policy_opt.step()

                value = self.get_value_operator()(minibatch)["state_value"]
                value_loss = nn.functional.mse_loss(value, minibatch["value_target"])
                self.value_opt.zero_grad()
                value_loss.backward()
                self.value_opt.step()

        train_info["reward"] = batch["reward"].detach().mean().item()
        
        return train_info
    
    def make_dataset(self, batch: TensorDictBase, num_minibatches: int):
        # batch = batch.view(-1)
        batch = {k: v.flatten(end_dim=1) for k, v in batch.items()}
        perm = torch.randperm(batch["advantage"].shape[0]).reshape(num_minibatches, -1)
        for indices in perm:
            yield {k: v[indices] for k, v in batch.items()}

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

        self.agents = self.envs.agent_types
        if len(self.agents) > 1:
            self.num_agents = {
                agent: getattr(self.envs, f"num_{agent}s") for agent in self.agents
            }
        else:
            self.num_agents = {self.agents[0]: self.envs.num_agents}
        self.num_steps = all_args.num_steps

        envs: MultiAgentVecTask = config["envs"]

        self.policies: Dict[str, ActorCriticPolicy] = {}
        for agent_type in self.agents:
            self.policies[agent_type] = ActorCriticPolicy(
                obs_spec=envs.observation_spec[f"obs@{agent_type}"],
                act_spec=envs.action_spec,
                cfg=Namespace(**cfg.params),
            )
            self.policies[agent_type].to(self.device)
        
        # timers & counters
        self.env_step_time = 0
        self.inf_step_time = 0
        
        self.total_env_steps = 0
        self.env_steps_this_run = 0
        self.total_episodes = 0
        self.episodes_this_run = 0

        if self.use_cl:
            self.task_difficulty_collate_fn = lambda episode_infos: episode_infos["reward_capture"].mean(-1)

    def run(self):
        self.training = True
        tensordict = self.envs.reset() # {agent:{obs}}

        common_keys = ["state", "next_state"]

        episode_infos = defaultdict(list)
        
        def joint_policy(tensordict: TensorDictBase) -> TensorDictBase:
            for agent_type, policy in self.policies.items():
                agent_td = tensordict.select(*common_keys)
                agent_td.update({k[:k.index("@")]: v for k ,v in tensordict.items() if k.endswith(f"@{agent_type}")})
                output_td = policy.forward(agent_td)
                for k in list(output_td.keys()):
                    if k not in common_keys:
                        output_td.rename_key(k, f"{k}@{agent_type}")
                tensordict.update(output_td)    
            return tensordict        
        
        replay_buffer = {}
        for agent_type in self.agents:
            replay_buffer[f"obs@{agent_type}"] = torch.zeros(self.num_steps, self.num_envs, self.num_agents[agent_type], *self.envs.observation_spec[f"obs@{agent_type}"].shape, device=self.device)
            # replay_buffer[f"next_obs@{agent_type}"] = torch.zeros(self.num_steps, self.num_envs, self.num_agents[agent_type], *self.envs.observation_spec[f"obs@{agent_type}"].shape, device=self.device)
            replay_buffer[f"actions@{agent_type}"] = torch.zeros(self.num_steps, self.num_envs, self.num_agents[agent_type], *self.envs.action_spec.shape, device=self.device)
            replay_buffer[f"reward@{agent_type}"] = torch.zeros(self.num_steps, self.num_envs, self.num_agents[agent_type], 3, device=self.device)
            replay_buffer[f"done@{agent_type}"] = torch.zeros(self.num_steps, self.num_envs, self.num_agents[agent_type], 1, device=self.device)
            replay_buffer[f"sample_log_prob@{agent_type}"] = torch.zeros(self.num_steps, self.num_envs, self.num_agents[agent_type], 1, device=self.device)
            replay_buffer[f"state_value@{agent_type}"] = torch.zeros(self.num_steps, self.num_envs, self.num_agents[agent_type], 1, device=self.device)
        replay_buffer["state"] = torch.zeros_like(replay_buffer["obs@predator"], device=self.device)
        replay_buffer["next_state"] = torch.zeros_like(replay_buffer["obs@predator"], device=self.device)
        
        def step_callback(env: MultiAgentVecTask, tensordict: TensorDictBase, step: int):  
            for k, v in replay_buffer.items():
                v[step] = tensordict[k]

            env_done = tensordict["env_done"].squeeze(-1)
            if env_done.any():
                for k, v in env.extras["episode"][env_done].items():
                    episode_infos[k].extend(v.tolist())

        for iteration in range(self.max_iterations):
            iter_start = time.time()
            for agent_type, policy in self.policies.items():
                policy.eval()
            with torch.no_grad(), set_exploration_mode("random"):
                tensordict = self.envs.rollout(max_steps=self.num_steps, policy=joint_policy, tensordict=tensordict, callback=step_callback)
            
            batch = replay_buffer
            for agent_type, policy in self.policies.items():
                agent_batch = {}
                agent_batch.update({k[:k.index("@")]: v.flatten(1, 2) for k, v in batch.items() if k.endswith(f"@{agent_type}")})
                agent_batch.update({k: v.flatten(1, 2) for k, v in batch.items() if k in common_keys})
                
                agent_train_info = policy.update(agent_batch)

            iter_end = time.time()
            if iteration % self.log_interval == 0:
                logging.info(f"FPS: {self.num_envs*self.num_steps/(iter_end-iter_start)}")
                collect_episode_infos(episode_infos, "train")
                episode_infos.clear()
            self.total_env_steps += self.num_steps * self.num_envs
        
        raise

        distributions = defaultdict(list)
        metric = "success"
        if "best_" + metric not in wandb.run.summary.keys():
            wandb.run.summary["best_" + metric] = 0.0

        # setup CL
        if self.use_cl:
            tasks = TaskDist(capacity=16384)
            def sample_task(envs, env_ids, env_states: torch.Tensor):
                # z = (1-self.env_steps_this_run / self.max_env_steps)*0.7
                z = 0.3
                if self.training and len(tasks) > 0:
                    p = np.random.rand()
                    if p < z:
                        sampled_tasks = tasks.sample(len(env_ids), "easy")
                        if sampled_tasks is not None:
                            envs.set_tasks(env_ids, sampled_tasks, env_states)
                    elif p < 0.7:
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
                    episode_info = infos["episode"][env_dones]
                    task_config: TensorDict = infos["task_config"][env_dones]
                    for k, v in episode_info.items():
                        episode_infos[k].extend(v.tolist())

                    if self.use_cl:
                        metrics = self.task_difficulty_collate_fn(episode_info)
                        tasks.add(task_config=task_config, metrics=metrics)

                    for k, v in task_config.items():
                        if v.squeeze(-1).dim() == 1: # e.g., target_speed
                            distributions[k].extend(v.squeeze(-1).tolist())

                    self.total_episodes += env_dones.sum()

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

            if iteration % self.eval_interval == 0:
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

    def restore(self, reset_steps=False):
        logging.info(f"Restoring models from {wandb.run.dir}")
        checkpoint = torch.load(os.path.join(wandb.run.dir, "checkpoint.pt"))
        for agent, policy in self.policies.items():
            policy.actor.load_state_dict(checkpoint[agent]["actor"])
            policy.critic.load_state_dict(checkpoint[agent]["critic"])
        if not reset_steps:
            self.total_episodes = checkpoint["episodes"]
            self.total_env_steps = checkpoint["env_steps"]

    def eval(self, eval_episodes, log=True):
        stamp_str = f"Eval at {self.total_env_steps}(total)/{self.env_steps_this_run}(this run) steps."
        logging.info(stamp_str)
        for agent, policy in self.policies.items():
            policy.actor.eval()
            policy.critic.eval()

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
        scatter_plot_data = defaultdict(lambda: [])
        metric = "reward_capture"

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
                    episode_info = infos["episode"][valid_envs]
                    task_config = infos["task_config"][valid_envs]

                    for k, v in episode_info.items():
                        episode_infos[k].extend(v.tolist())

                    for k, v in task_config.items():
                        if v.squeeze(-1).dim() == 1:
                            episode_metric = episode_info[metric]
                            if episode_metric.dim() > 1: 
                                episode_metric = episode_metric.mean(-1)
                            scatter_plot_data[metric].extend(episode_metric.tolist())
                            scatter_plot_data[k].extend(v.squeeze(-1).tolist()) # (num_envs, 1) -> List
                    
                    episode_count += valid_envs.sum()
                already_reset |= env_dones

            for buffer in self.buffers.values(): buffer.after_update()

        eval_infos = {"env_step": self.total_env_steps}
        eval_infos.update(collect_episode_infos(episode_infos, "eval"))
        
        if len(scatter_plot_data) == 2:
            data = list(zip(*scatter_plot_data.values()))
            keys = list(scatter_plot_data.keys())
            table = wandb.Table(data=data, columns=keys)
            name = "eval/" + " vs ".join(keys)
            eval_infos[name] = wandb.plot.scatter(table, *keys, title=name)
        
        if log:
            self.log(eval_infos)

# class TaskDist:
#     def __init__(self, capacity: int=1000, easy_threshold: float=400, hard_threshold:float=200) -> None:
#         self.capacity = capacity
#         self.task_config = None
#         self.metrics = None
#         self.easy_threshold = easy_threshold
#         self.hard_threshold = hard_threshold
    
#     def add(self, task_config: TensorDict, metrics: torch.Tensor) -> None:
#         if self.task_config is None:
#             self.task_config = task_config
#         else:
#             self.task_config = torch.cat((self.task_config, task_config))[-self.capacity:]
#         if self.metrics is None:
#             self.metrics = metrics[-self.capacity:]
#         else:
#             self.metrics = torch.cat([self.metrics, metrics])[-self.capacity:]
    
#     def sample(self, n: int, mode: str="easy") -> Union[TensorDict, None]:
#         if mode == "easy":
#             task_config = self.task_config[self.metrics>=self.easy_threshold]
#         elif mode == "hard":
#             task_config = self.task_config[self.metrics<self.hard_threshold]
#         elif mode == "medium":
#             task_config = self.task_config[(self.metrics>=self.hard_threshold) & (self.metrics<self.easy_threshold)]
#         else:
#             raise ValueError(f"Unknown mode: {mode}")

#         if task_config.batch_size[0] > 0:
#             return task_config[torch.randint(0, task_config.batch_size[0], (n,))]
#         else:
#             return None
        
#     def __len__(self) -> int:
#         return 0 if self.metrics is None else len(self.metrics)
