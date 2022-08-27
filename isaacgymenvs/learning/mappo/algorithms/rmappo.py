from ast import Not
from collections import defaultdict
import functools
from torchrl.data.tensordict.tensordict import TensorDict
from .r_actor_critic import R_Actor, R_Critic
import gym
import torch
import torch.nn as nn
from dataclasses import dataclass
from torch.nn.functional import mse_loss, huber_loss
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
        self._use_max_grad_norm = cfg.use_max_grad_norm
        self._use_clipped_value_loss = cfg.use_clipped_value_loss
        self._use_huber_loss = cfg.use_huber_loss
        self._use_popart = cfg.use_popart
        self._use_valuenorm = cfg.use_valuenorm
        self._use_value_active_masks = cfg.use_value_active_masks
        self._use_policy_active_masks = cfg.use_policy_active_masks

        self.num_critics = cfg.num_critics
        # policy models
        self.actor = R_Actor(cfg, obs_space, act_space, self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr, weight_decay=cfg.weight_decay)
        
        self.critic_loss_fn = functools.partial(huber_loss, delta=self.huber_delta) if self._use_huber_loss else mse_loss
        self.critics = [R_Critic(cfg, state_space, self.device) for _ in range(self.num_critics)]
        self.critic_optimizers = [torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr, weight_decay=cfg.weight_decay) for critic in self.critics]

        # self.scaler = torch.cuda.amp.GradScaler()

        assert (self._use_popart and self._use_valuenorm) == False, ("self._use_popart and self._use_valuenorm can not be set True simultaneously")
        
        if self._use_popart:
            self.value_normalizer = self.policy.critic.v_out
            if self._use_policy_vhead:
                self.policy_value_normalizer = self.policy.actor.v_out
        elif self._use_valuenorm:
            self.value_normalizer = ValueNorm(1, device = self.device)

    def get_action_and_value(self, 
            share_obs, obs, 
            rnn_states_actor=None, 
            rnn_states_critic=None, 
            masks=None, 
            available_actions=None, 
            deterministic=False) -> Dict[str, torch.Tensor]:
        actions, action_log_probs, rnn_states_actor = self.actor(obs, rnn_states_actor, masks, available_actions, deterministic)
        values, rnn_states_critic = self.critics[0](share_obs, rnn_states_critic, masks)
        return {
            "value": values, 
            "action": actions, 
            "action_log_prob": action_log_probs, 
            "rnn_state_actor": rnn_states_actor, 
            "rnn_state_critic": rnn_states_critic
        }

    def get_value(self, share_obs, rnn_state_critic, mask) -> torch.Tensor:
        values, critic_states = self.critics[0](share_obs, rnn_state_critic, mask)
        return values

    def act(self, 
            obs, 
            rnn_states_actor=None, 
            masks=None, available_actions=None, deterministic=False):
        actions, _, rnn_states_actor = self.actor(obs, rnn_states_actor, masks, available_actions, deterministic)
        return actions, rnn_states_actor
    
    def update_actor(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        action_log_probs, dist_entropy = self.actor.evaluate_actions(
            batch["obs"], 
            batch["rnn_state_actor"], 
            batch["action"], 
            batch["mask"], 
        )
        ratio = torch.exp(action_log_probs - batch["old_action_log_prob"])

        surr1 = ratio * batch["advantage"]
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * batch["advantage"]

        policy_loss = - torch.min(surr1, surr2).mean()
        policy_loss = policy_loss - self.entropy_coef * dist_entropy
        
        policy_loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()
        self.actor_optimizer.zero_grad(set_to_none=True)
        return {"policy_loss": policy_loss.item(), "ratio": ratio.detach().mean().item(), "actor_grad_norm": grad_norm.item()}

    def calc_value_loss(self, critic: R_Critic, batch: Dict[str, torch.Tensor]):
        value = critic.get_value(batch["state"], batch["rnn_state_critic"], batch["mask"])
        value_pred_clipped = batch["value_pred"] + (value - batch["value_pred"]).clamp(-self.clip_param, self.clip_param)
        return_batch = batch["return"]
        if self._use_popart or self._use_valuenorm:
            self.value_normalizer.update(return_batch)
            return_batch = self.value_normalizer.normalize(return_batch)

        value_loss_clipped = self.critic_loss_fn(return_batch, value_pred_clipped)
        value_loss_original = self.critic_loss_fn(return_batch, value)
        
        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original
        return value, value_loss

    def update_critic(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        values = []
        result_dict = {}
        for i, (critic, optimizer) in enumerate(zip(self.critics, self.critic_optimizers)):
            value, value_loss = self.calc_value_loss(critic, batch)
            values.append(value)
            (value_loss * self.value_loss_coef).backward()
            grad_norm = nn.utils.clip_grad_norm_(critic.parameters(), self.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            result_dict[f"value_loss_{i}"] = value_loss.item()
            result_dict[f"critic_grad_norm_{i}"] = grad_norm.item()

        if len(values) > 1:
            result_dict["value_std"] = torch.concat(values, dim=-1).std(dim=-1).mean()
        return result_dict

    def ppo_update(self, sample):

        state_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
        value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
        adv_targ, available_actions_batch = sample        

        result_dict = {}
        actor_batch = {"obs": obs_batch, "rnn_state_actor": rnn_states_batch, "action": actions_batch, "mask": masks_batch, "old_action_log_prob": old_action_log_probs_batch, "advantage": adv_targ}
        critic_batch = {"state": state_batch, "value_pred": value_preds_batch, "rnn_state_critic": rnn_states_critic_batch, "mask": masks_batch, "return": return_batch}
        result_dict.update(self.update_actor(actor_batch))
        result_dict.update(self.update_critic(critic_batch))

        return result_dict
        
    def train(self, buffer, turn_on=True):
        if self._use_popart or self._use_valuenorm:
            advantages: torch.Tensor = buffer.returns[:-1] - self.value_normalizer.denormalize(buffer.value_preds[:-1])
        else:
            advantages: torch.Tensor = buffer.returns[:-1] - buffer.value_preds[:-1]
        advantages[buffer.active_masks[:-1] == 0.0].zero_()
        advantages = (advantages - advantages.mean())  / (advantages.std() + 1e-8)

        train_info = defaultdict(float)

        for _ in range(self.ppo_epoch):
            if self._use_recurrent_policy:
                data_generator = buffer.recurrent_generator(advantages, self.num_mini_batch, self.data_chunk_length)
            else:
                data_generator = buffer.feed_forward_generator(advantages, self.num_mini_batch)

            for i, sample in enumerate(data_generator):
                result_dict = self.ppo_update(sample)
                for k, v in result_dict.items():
                    train_info[k] += v

        num_updates = self.ppo_epoch * self.num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates

        return train_info

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
            