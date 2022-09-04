from collections import defaultdict
import functools
from .r_actor_critic import R_Actor, R_Critic
import gym
import torch
import torch.nn as nn
from dataclasses import dataclass
from torch.nn.functional import mse_loss, huber_loss
import math
from isaacgymenvs.learning.mappo.utils.valuenorm import ValueNorm
from isaacgymenvs.learning.mappo.utils.data import TensorDict
from typing import Any, Dict, Tuple
import time

@dataclass
class MAPPOPolicyConfig:
    pass
#     # ppo
#     ppo_epoch: int
#     clip_param: 
#     num_mini_batch:
#     data_chunk_length:

#     # actor

#     # optimizers
#     actor_lr: float
#     critic_lr: float
#     weight_decay: float
#     device: torch.device

def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class MAPPOPolicy:
    def __init__(self, 
            cfg: MAPPOPolicyConfig,
            num_agents: int,
            obs_space: gym.Space,
            state_space: gym.Space,
            act_space: gym.Space) -> None:
        cfg.device = "cuda"
        cfg.actor_lr = cfg.actor_lr
        self.device = cfg.device

        # ppo
        self.clip_param = cfg.clip_param
        self.ppo_epoch = cfg.ppo_epochs
        self.num_mini_batch = cfg.num_minibatches
        self.data_chunk_length = cfg.data_chunk_length
        self.policy_value_loss_coef = cfg.policy_value_loss_coef
        self.value_loss_coef = cfg.value_loss_coef
        self.entropy_coef = cfg.entropy_coef
        self.max_grad_norm = cfg.max_grad_norm       
        self.huber_delta = cfg.huber_delta

        self._use_recurrent_policy = cfg.use_recurrent_policy
        self._use_max_grad_norm = cfg.use_max_grad_norm
        self._use_clipped_value_loss = cfg.use_clipped_value_loss
        self._use_huber_loss = cfg.use_huber_loss
        self._use_popart = cfg.use_popart
        self._use_valuenorm = cfg.use_valuenorm
        self._use_value_active_masks = cfg.use_value_active_masks
        self._use_policy_active_masks = cfg.use_policy_active_masks

        self.num_agents = num_agents
        self.num_critics = cfg.num_critics
        # policy models
        self.action_dim = act_space.shape[0]
        self.actor = R_Actor(cfg, obs_space, act_space, self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr, weight_decay=cfg.weight_decay)
        
        self.critic_loss_fn = functools.partial(huber_loss, delta=self.huber_delta) if self._use_huber_loss else mse_loss
        self.critics = [R_Critic(cfg, state_space, self.device) for _ in range(self.num_critics)]
        self.critic_optimizers = [torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr, weight_decay=cfg.weight_decay) for critic in self.critics]

        # self.scaler = torch.cuda.amp.GradScaler()

        assert (self._use_popart and self._use_valuenorm) == False, ("self._use_popart and self._use_valuenorm can not be set True simultaneously")
        
        if self._use_popart:
            self.value_normalizer = self.critic.v_out
        elif self._use_valuenorm:
            self.value_normalizer = ValueNorm(1, device = self.device)

    def __call__(self, tensordict: TensorDict, **kwargs):
        result_dict = TensorDict()
        result_dict.update(self.policy_op(tensordict, **kwargs))
        result_dict.update(self.value_op(tensordict, **kwargs))
        return result_dict
        
    def policy_op(self, tensordict: TensorDict, deterministic=False):
        tensordict = tensordict.flatten(0, 1)
        actions, action_log_porobs, rnn_states_actor = self.actor(
            tensordict["obs"], 
            tensordict["rnn_states_actor"], 
            tensordict["masks"], 
            tensordict["available_actions"], deterministic=deterministic)
        result_dict = TensorDict({"actions": actions, "action_log_probs": action_log_porobs})
        return result_dict.unflatten(0, (-1, self.num_agents))

    def value_op(self, tensordict: TensorDict, disagreement=False) -> TensorDict:
        """
        input: {
            obs,
            rnn_states_critic,
            masks,
        }
        return: {
            values,
            value_stds,
            next_rnn_states_critic,
        }
        """
        result_dict = TensorDict()
        values = []
        for i, critic in enumerate(self.critics):
            value, rnn_state_critic = critic(
                tensordict["obs"], 
                None, #tensordict["rnn_state_critic"], 
                tensordict["mask"])
            values.append(value)
        values = torch.stack(values, 0)
        result_dict["values"] = values.mean(0)
        if disagreement:
            result_dict["value_stds"] = values.std(0)
        return TensorDict(result_dict)
    
    def update_actor(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        action_log_probs, dist_entropy = self.actor.evaluate_actions(
            batch["obs"], 
            batch["rnn_states_actor"], 
            batch["actions"], 
            1.0 - batch["dones"], 
        )
        ratio = torch.exp(action_log_probs - batch["action_log_probs"])

        surr1 = ratio * batch["advantages"]
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * batch["advantages"]

        policy_loss = - torch.min(surr1, surr2).mean() * self.action_dim
        policy_loss = policy_loss - self.entropy_coef * dist_entropy
        
        policy_loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()
        self.actor_optimizer.zero_grad(set_to_none=True)
        return {"policy_loss": policy_loss.item(), "ratio": ratio.detach().mean().item(), "actor_grad_norm": grad_norm.item(), "dist_entropy": dist_entropy.item()}

    def calc_value_loss(self, critic: R_Critic, batch: Dict[str, torch.Tensor]):
        values = critic.get_value(batch["obs"], batch["rnn_states_critic"], 1.0-batch["dones"])
        value_pred_clipped = batch["values"] + (values - batch["values"]).clamp(-self.clip_param, self.clip_param)
        return_batch = batch["returns"]
        if self._use_popart or self._use_valuenorm:
            self.value_normalizer.update(return_batch)
            return_batch = self.value_normalizer.normalize(return_batch)

        value_loss_clipped = self.critic_loss_fn(return_batch, value_pred_clipped)
        value_loss_original = self.critic_loss_fn(return_batch, values)
        
        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original
        return values, value_loss

    def update_critic(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        all_values = []
        result_dict = {}
        for i, (critic, optimizer) in enumerate(zip(self.critics, self.critic_optimizers)):
            values, value_loss = self.calc_value_loss(critic, batch)
            all_values.append(values)
            (value_loss * self.value_loss_coef).backward()
            grad_norm = nn.utils.clip_grad_norm_(critic.parameters(), self.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            result_dict[f"value_loss_{i}"] = value_loss.item()
            result_dict[f"critic_grad_norm_{i}"] = grad_norm.item()

        if len(all_values) > 1:
            result_dict["value_std"] = torch.concat(all_values, dim=-1).std(dim=-1).mean()
        return result_dict

    def train_on_batch(self, batch: TensorDict):
        with torch.no_grad():
            value_output = self.value_op(batch[[-1]].step())
        
        values = batch["values"]
        next_value = value_output["values"].squeeze(0)
        if self._use_valuenorm or self._use_popart:
            values = self.value_normalizer.denormalize(values)
            next_value = self.value_normalizer.denormalize(next_value)
        batch["advantages"], batch["returns"] = compute_gae(
            rewards=batch["rewards"], 
            dones=batch["dones"],
            next_done=batch["next_dones"][-1],
            values=values,
            next_value=next_value)

        train_info = defaultdict(float)
        for ppo_epoch in range(self.ppo_epoch):
            for minibatch in self.make_dataset(batch):
                for k, v in {**self.update_actor(minibatch), **self.update_critic(minibatch)}.items():
                    train_info[k] += v

        num_updates = self.ppo_epoch * self.num_mini_batch
        for k in train_info.keys():
            train_info[k] /= num_updates
        
        return train_info

    def make_dataset(self, tensordict: TensorDict):
        """
            tensordict: [T, E, A]
        """
        tensordict = tensordict.flatten(0, 2)
        batch_size = tensordict["obs"].shape[0]
        perm = torch.randperm(batch_size).reshape(self.num_mini_batch, -1)
        for indices in perm:
            yield tensordict[indices]

    def make_recurrent_dataset(tensordict):
        """
        
        """
        raise NotImplementedError

    def prep_training(self):
        self.actor.train()
        [critic.train() for critic in self.critics]

    def prep_rollout(self):
        self.actor.eval()
        [critic.eval() for critic in self.critics]
    
    def state_dict(self):
        return {
            "actor": self.actor.state_dict(),
            "critics": [critic.state_dict() for critic in self.critics]
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.actor.load_state_dict(state_dict["actor"])
        for critic, critic_state_dict in zip(self.critics, state_dict["critics"]):
            critic.load_state_dict(critic_state_dict)
            
def compute_gae(
        rewards: torch.Tensor, # [T, N, ...]
        dones: torch.Tensor, # [T, N, ...]
        next_done: torch.Tensor, # [N, ...] 
        values: torch.Tensor, 
        next_value: torch.Tensor, 
        gamma=0.99, lmbda=0.95, normalize_advantages=True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
        [T, ...]
    """
    num_steps = rewards.size(0)
    next_mask = 1.0 - next_done
    gae = 0
    advantages = torch.zeros_like(rewards)
    for step in reversed(range(num_steps)):

        delta = rewards[step] + gamma*next_value*next_mask - values[step]
        advantages[step] = gae = delta + gamma*lmbda*next_mask * gae
        
        next_value = values[step]
        next_mask = 1.0 - dones[step]
    returns = advantages + values # aka. value targets
    if normalize_advantages:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return advantages, returns