from ast import Not
from torchrl.data.tensordict.tensordict import TensorDict
from .r_actor_critic import R_Actor, R_Critic
import gym
import torch
import torch.nn as nn
from dataclasses import dataclass
from torch.nn.functional import mse_loss
import math
from isaacgymenvs.learning.mappo.utils.valuenorm import ValueNorm
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

def huber_loss(e, d):
    a = (abs(e) <= d).float()
    b = (e > d).float()
    return a*e**2/2 + b*d*(abs(e)-d/2)

def get_gard_norm(it):
    sum_grad = 0
    for x in it:
        if x.grad is None:
            continue
        sum_grad += x.grad.norm() ** 2
    return math.sqrt(sum_grad)

class MAPPOPolicy:
    def __init__(self, 
            cfg: MAPPOPolicyConfig,
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
        self._use_naive_recurrent = cfg.use_naive_recurrent_policy
        self._use_max_grad_norm = cfg.use_max_grad_norm
        self._use_clipped_value_loss = cfg.use_clipped_value_loss
        self._use_huber_loss = cfg.use_huber_loss
        self._use_popart = cfg.use_popart
        self._use_valuenorm = cfg.use_valuenorm
        self._use_value_active_masks = cfg.use_value_active_masks
        self._use_policy_active_masks = cfg.use_policy_active_masks
        self._use_policy_vhead = cfg.use_policy_vhead

        # policy models
        self.actor = R_Actor(cfg, obs_space, act_space, self.device)
        self.critic = R_Critic(cfg, state_space, self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr, weight_decay=cfg.weight_decay)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr, weight_decay=cfg.weight_decay)

        assert (self._use_popart and self._use_valuenorm) == False, ("self._use_popart and self._use_valuenorm can not be set True simultaneously")
        
        if self._use_popart:
            self.value_normalizer = self.policy.critic.v_out
            if self._use_policy_vhead:
                self.policy_value_normalizer = self.policy.actor.v_out
        elif self._use_valuenorm:
            self.value_normalizer = ValueNorm(1, device = self.device)
            if self._use_policy_vhead:
                self.policy_value_normalizer = ValueNorm(1, device = self.device)
        else:
            self.value_normalizer = None
            if self._use_policy_vhead:
                self.policy_value_normalizer = None

    def __call__(self, tensordict: TensorDict, compute_values) -> TensorDict:
        actions, action_log_probs, rnn_states_actor = self.actor(
            tensordict["obs"], 
            tensordict.get("rnn_states_actor", None), 
            tensordict["masks"], 
            tensordict.get("available_actions", None))
        tensordict.update({"actions": actions, "action_log_probs": action_log_probs})
        if rnn_states_actor is not None:
            tensordict.set("next_rnn_states_actor", rnn_states_actor)
        if compute_values:
            values, rnn_states_critic = self.critic(
                tensordict["state"], 
                tensordict.get("rnn_states_critic", None), 
                tensordict["masks"])
            tensordict.set("values", values)
            if rnn_states_critic is not None:
                tensordict.set("next_rnn_states_critic", rnn_states_critic)
        return tensordict

    def lr_decay(self, episode, episodes):
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)

    def get_action_and_value(self, 
            share_obs, obs, 
            rnn_states_actor=None, 
            rnn_states_critic=None, 
            masks=None, 
            available_actions=None, 
            deterministic=False) -> Dict[str, torch.Tensor]:
        actions, action_log_probs, rnn_states_actor = self.actor(obs, rnn_states_actor, masks, available_actions, deterministic)
        values, rnn_states_critic = self.critic(share_obs, rnn_states_critic, masks)
        return {
            "value": values, 
            "action": actions, 
            "action_log_prob": action_log_probs, 
            "rnn_state_actor": rnn_states_actor, 
            "rnn_state_critic": rnn_states_critic
        }

    def get_values(self, share_obs, rnn_states_critic, masks) -> torch.Tensor:
        values, critic_states = self.critic(share_obs, rnn_states_critic, masks)
        return values

    def evaluate_actions(self, share_obs, obs, rnn_states_actor, rnn_states_critic, action, masks, available_actions=None, active_masks=None):
        action_log_probs, dist_entropy, policy_values = self.actor.evaluate_actions(obs, rnn_states_actor, action, masks, available_actions, active_masks)
        values, _ = self.critic(share_obs, rnn_states_critic, masks)
        return values, action_log_probs, dist_entropy, policy_values

    def act(self, obs, rnn_states_actor, masks, available_actions=None, deterministic=False):
        actions, _, rnn_states_actor = self.actor(obs, rnn_states_actor, masks, available_actions, deterministic)
        return actions, rnn_states_actor
    
    def cal_value_loss(self, value_normalizer, values, value_preds_batch, return_batch, active_masks_batch):
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
        

        if self._use_huber_loss:
            if self._use_popart or self._use_valuenorm:
                value_normalizer.update(return_batch)
                error_clipped = value_normalizer.normalize(return_batch) - value_pred_clipped
                error_original = value_normalizer.normalize(return_batch) - values
            else:
                error_clipped = return_batch - value_pred_clipped
                error_original = return_batch - values
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            if self._use_popart or self._use_valuenorm:
                value_loss_clipped = mse_loss(value_normalizer.normalize(return_batch), value_pred_clipped)
                value_loss_original = mse_loss(value_normalizer.normalize(return_batch), values)
            else:
                value_loss_clipped = mse_loss(return_batch, value_pred_clipped)
                value_loss_original = mse_loss(return_batch, values)

        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        if self._use_value_active_masks:
            value_loss = (value_loss * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            value_loss = value_loss.mean()

        return value_loss

    def ppo_update(self, sample, turn_on=True):

        share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
        value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
        adv_targ, available_actions_batch = sample

        # Reshape to do in a single forward pass for all steps
        values, action_log_probs, dist_entropy, policy_values = self.evaluate_actions(share_obs_batch,
                                                                              obs_batch, 
                                                                              rnn_states_batch, 
                                                                              rnn_states_critic_batch, 
                                                                              actions_batch, 
                                                                              masks_batch, 
                                                                              available_actions_batch,
                                                                              active_masks_batch)
        # actor update
        ratio = torch.exp(action_log_probs - old_action_log_probs_batch)

        surr1 = ratio * adv_targ
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ

        if self._use_policy_active_masks:
            policy_action_loss = (-torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True) * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        if self._use_policy_vhead:
            policy_value_loss = self.cal_value_loss(self.policy_value_normalizer, policy_values, value_preds_batch, return_batch, active_masks_batch)       
            policy_loss = policy_action_loss + policy_value_loss * self.policy_value_loss_coef
        else:
            policy_loss = policy_action_loss

        self.actor_optimizer.zero_grad()

        if turn_on:
            (policy_loss - dist_entropy * self.entropy_coef).backward()

        if self._use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        else:
            actor_grad_norm = get_gard_norm(self.actor.parameters())

        self.actor_optimizer.step()

        # critic update
        value_loss = self.cal_value_loss(self.value_normalizer, values, value_preds_batch, return_batch, active_masks_batch)

        self.critic_optimizer.zero_grad()

        (value_loss * self.value_loss_coef).backward()

        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        else:
            critic_grad_norm = get_gard_norm(self.critic.parameters())

        self.critic_optimizer.step()

        return value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, ratio
        
    def train(self, buffer, turn_on=True):
        if self._use_popart or self._use_valuenorm:
            advantages: torch.Tensor = buffer.returns[:-1] - self.value_normalizer.denormalize(buffer.value_preds[:-1])
        else:
            advantages: torch.Tensor = buffer.returns[:-1] - buffer.value_preds[:-1]
        advantages[buffer.active_masks[:-1] == 0.0].zero_()
        advantages = (advantages - advantages.mean())  / (advantages.std() + 1e-8)

        train_info = {}

        train_info['value_loss'] = 0
        train_info['policy_loss'] = 0
        train_info['dist_entropy'] = 0
        train_info['actor_grad_norm'] = 0
        train_info['critic_grad_norm'] = 0
        train_info['ratio'] = 0

        for _ in range(self.ppo_epoch):
            if self._use_recurrent_policy:
                data_generator = buffer.recurrent_generator(advantages, self.num_mini_batch, self.data_chunk_length)
            elif self._use_naive_recurrent:
                data_generator = buffer.naive_recurrent_generator(advantages, self.num_mini_batch)
            else:
                data_generator = buffer.feed_forward_generator(advantages, self.num_mini_batch)

            for i, sample in enumerate(data_generator):
                value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, ratio \
                    = self.ppo_update(sample, turn_on)

                train_info['value_loss'] += value_loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['dist_entropy'] += dist_entropy.item()
                
                if int(torch.__version__[2]) < 5:
                    train_info['actor_grad_norm'] += actor_grad_norm
                    train_info['critic_grad_norm'] += critic_grad_norm
                else:
                    train_info['actor_grad_norm'] += actor_grad_norm.item()
                    train_info['critic_grad_norm'] += critic_grad_norm.item()

                train_info['ratio'] += ratio.mean().item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates

        train_info["advantages"] = advantages.mean()

        return train_info

    def prep_training(self):
        self.actor.train()
        self.critic.train()

    def prep_rollout(self):
        self.actor.eval()
        self.critic.eval()
    
    def get_initial_rnn_states(self, *batch_size):
        if self._use_recurrent_policy or self._use_naive_recurrent:
            return (
                torch.zeros(self.actor._recurrent_N, *batch_size,  self.actor.hidden_size, device=self.device),
                torch.zeros(self.critic._recurrent_N, *batch_size, self.actor.hidden_size, device=self.device)
            )
        else:
            return (None, None)