import time
from typing import Sequence, Tuple
import torch
import numpy as np
import gym
from collections import defaultdict

from ..utils.util import check, get_shape_from_obs_space, get_shape_from_act_space
from gym import spaces


def create_buffer(space: spaces.Space, base_shape: Tuple[int, ...], device="cuda"):
    if isinstance(space, spaces.Dict):
        return {k: create_buffer(v) for k, v in space.items()}
    elif isinstance(space, spaces.Box):
        return torch.zeros(base_shape+space.shape, device=device)
    elif isinstance(space, spaces.Discrete):
        return torch.zeros(base_shape+(space.n, ), device=device)
    elif isinstance(space, spaces.MultiDiscrete):
        return torch.zeros(base_shape+tuple(space.nvec), device=device)
    elif isinstance(space, list):
        return torch.zeros(base_shape+(space[0],), device=device)
    else:
        raise TypeError(f"Unsupported space type: {type(space)}")


class SharedReplayBuffer(object):
    def __init__(self, args,
                 num_envs: int,
                 num_agents: int,
                 obs_space: gym.Space,
                 share_obs_space: gym.Space,
                 act_space: gym.Space,
                 device="cuda"):

        self.num_steps = args.num_steps
        self.num_envs = num_envs
        self.num_agents = num_agents

        self.hidden_size = args.hidden_size
        self.recurrent_N = args.recurrent_N
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        self._use_gae = args.use_gae
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self._use_proper_time_limits = args.use_proper_time_limits

        self._mixed_obs = False  # for mixed observation

        obs_shape = get_shape_from_obs_space(obs_space)
        share_obs_shape = get_shape_from_obs_space(share_obs_space)

        batch_size = self.num_envs * self.num_agents
        base_shape = (self.num_steps + 1, self.num_envs, self.num_agents)

        self.obs = create_buffer(obs_space, base_shape, device)
        self.share_obs = create_buffer(share_obs_space, base_shape, device)

        self.rnn_states = torch.zeros(
            (self.num_steps + 1, self.recurrent_N, batch_size, self.hidden_size), device=device)
        self.rnn_states_critic = torch.zeros_like(self.rnn_states)

        self.value_preds = torch.zeros(
            (self.num_steps + 1, self.num_envs, num_agents, 1), device=device)
        self.returns = torch.zeros_like(self.value_preds)

        if act_space.__class__.__name__ == 'Discrete':
            self.available_actions = torch.ones(
                (self.num_steps + 1, self.num_envs, num_agents, act_space.n), device=device)
        else:
            self.available_actions = None

        act_shape = get_shape_from_act_space(act_space)

        self.actions = torch.zeros(
            (self.num_steps, self.num_envs, num_agents, act_shape), device=device)
        self.action_log_probs = torch.zeros(
            (self.num_steps, self.num_envs, num_agents, act_shape), device=device)
        self.rewards = torch.zeros(
            (self.num_steps, self.num_envs, num_agents, 1), device=device)

        self.masks = torch.ones(
            (self.num_steps + 1, self.num_envs, num_agents, 1), device=device)
        self.bad_masks = torch.ones_like(self.masks)
        self.active_masks = torch.ones_like(self.masks)

        self.step = 0

    def insert(self, 
            share_obs, 
            obs: torch.Tensor,
            actions, 
            action_log_probs,
            value_preds, 
            rewards, 
            masks, 
            rnn_states = None, 
            rnn_states_critic = None, 
            bad_masks=None, active_masks=None, 
            available_actions=None):
        
        self.share_obs[self.step + 1] = share_obs
        self.obs[self.step + 1] = obs

        if rnn_states is not None:
            self.rnn_states[self.step + 1] = rnn_states
        if rnn_states_critic is not None:
            self.rnn_states_critic[self.step + 1] = rnn_states_critic
        
        self.actions[self.step] = actions
        self.action_log_probs[self.step] = action_log_probs
        self.value_preds[self.step] = value_preds
        self.rewards[self.step] = rewards
        self.masks[self.step + 1] = masks

        if bad_masks is not None:
            self.bad_masks[self.step + 1] = bad_masks
        if active_masks is not None:
            self.active_masks[self.step + 1] = active_masks
        if available_actions is not None:
            self.available_actions[self.step + 1] = available_actions

        self.step = (self.step + 1) % self.num_steps

    def chooseinsert(self, share_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs,
                     value_preds, rewards, masks, bad_masks=None, active_masks=None, available_actions=None):
        self.share_obs[self.step] = share_obs
        self.obs[self.step] = obs
        self.rnn_states[self.step + 1] = rnn_states
        self.rnn_states_critic[self.step + 1] = rnn_states_critic
        self.actions[self.step] = actions
        self.action_log_probs[self.step] = action_log_probs
        self.value_preds[self.step] = value_preds
        self.rewards[self.step] = rewards
        self.masks[self.step + 1] = masks
        if bad_masks is not None:
            self.bad_masks[self.step + 1] = bad_masks
        if active_masks is not None:
            self.active_masks[self.step] = active_masks
        if available_actions is not None:
            self.available_actions[self.step] = available_actions

        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        if self._mixed_obs:
            for key in self.share_obs.keys():
                self.share_obs[key][0] = self.share_obs[key][-1]
            for key in self.obs.keys():
                self.obs[key][0] = self.obs[key][-1]
        else:
            self.share_obs[0] = self.share_obs[-1]
            self.obs[0] = self.obs[-1]
        self.rnn_states[0] = self.rnn_states[-1]
        self.rnn_states_critic[0] = self.rnn_states_critic[-1]
        self.masks[0] = self.masks[-1]
        self.bad_masks[0] = self.bad_masks[-1]
        self.active_masks[0] = self.active_masks[-1]
        if self.available_actions is not None:
            self.available_actions[0] = self.available_actions[-1]

    def chooseafter_update(self):
        self.rnn_states[0] = self.rnn_states[-1]
        self.rnn_states_critic[0] = self.rnn_states_critic[-1]
        self.masks[0] = self.masks[-1]
        self.bad_masks[0] = self.bad_masks[-1]

    def compute_returns(self, next_value, value_normalizer=None):
        if self._use_proper_time_limits:
            if self._use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.num_steps)):
                    if self._use_popart or self._use_valuenorm:
                        # step + 1
                        delta = self.rewards[step] + self.gamma * value_normalizer.denormalize(self.value_preds[step + 1]) * self.masks[step + 1]  \
                            - value_normalizer.denormalize(self.value_preds[step])
                        gae = delta + self.gamma * \
                            self.gae_lambda * gae * self.masks[step + 1]
                        gae = gae * self.bad_masks[step + 1]
                        self.returns[step] = gae + \
                            value_normalizer.denormalize(
                                self.value_preds[step])
                    else:
                        delta = self.rewards[step] + self.gamma * self.value_preds[step +
                                                                                   1] * self.masks[step + 1] - self.value_preds[step]
                        gae = delta + self.gamma * \
                            self.gae_lambda * self.masks[step + 1] * gae
                        gae = gae * self.bad_masks[step + 1]
                        self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.num_steps)):
                    if self._use_popart or self._use_valuenorm:
                        self.returns[step] = (self.returns[step + 1] * self.gamma * self.masks[step + 1] + self.rewards[step]) * self.bad_masks[step + 1] \
                            + (1 - self.bad_masks[step + 1]) * value_normalizer.denormalize(self.value_preds[step])
                    else:
                        self.returns[step] = (self.returns[step + 1] * self.gamma * self.masks[step + 1] + self.rewards[step]) * self.bad_masks[step + 1] \
                            + (1 - self.bad_masks[step + 1]
                               ) * self.value_preds[step]
        else:
            if self._use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.num_steps)):
                    if self._use_popart or self._use_valuenorm:
                        delta = self.rewards[step] + self.gamma * value_normalizer.denormalize(self.value_preds[step + 1]) * self.masks[step + 1] \
                            - value_normalizer.denormalize(self.value_preds[step])
                        gae = delta + self.gamma * \
                            self.gae_lambda * self.masks[step + 1] * gae
                        self.returns[step] = gae + \
                            value_normalizer.denormalize(
                                self.value_preds[step])
                    else:
                        delta = self.rewards[step] + self.gamma * self.value_preds[step +
                                                                                   1] * self.masks[step + 1] - self.value_preds[step]
                        gae = delta + self.gamma * \
                            self.gae_lambda * self.masks[step + 1] * gae
                        self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.shape[0])):
                    self.returns[step] = self.returns[step + 1] * \
                        self.gamma * self.masks[step + 1] + self.rewards[step]

    def feed_forward_generator(self, advantages, num_mini_batch=None, mini_batch_size=None):
        num_steps, num_envs, num_agents = self.rewards.shape[0:3]
        batch_size = num_envs * num_steps * num_agents

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) * number of agents ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(num_envs, num_steps, num_agents, num_envs * num_steps * num_agents,
                          num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch

        rand = torch.randperm(batch_size)
        sampler = [rand[i*mini_batch_size:(i+1)*mini_batch_size]
                   for i in range(num_mini_batch)]

        if self._mixed_obs:
            share_obs = {}
            obs = {}
            for key in self.share_obs.keys():
                share_obs[key] = self.share_obs[key][:-
                                                     1].reshape(-1, *self.share_obs[key].shape[3:])
            for key in self.obs.keys():
                obs[key] = self.obs[key][:-
                                         1].reshape(-1, *self.obs[key].shape[3:])
        else:
            share_obs = self.share_obs[:-
                                       1].reshape(-1, *self.share_obs.shape[3:])
            obs = self.obs[:-1].reshape(-1, *self.obs.shape[3:])
        

        actions = self.actions.reshape(-1, self.actions.shape[-1])
        if self.available_actions is not None:
            available_actions = self.available_actions[:-
                                                       1].reshape(-1, self.available_actions.shape[-1])
        value_preds = self.value_preds[:-1].reshape(-1, 1)
        returns = self.returns[:-1].reshape(-1, 1)
        masks = self.masks[:-1].reshape(-1, 1)
        active_masks = self.active_masks[:-1].reshape(-1, 1)
        action_log_probs = self.action_log_probs.reshape(
            -1, self.action_log_probs.shape[-1])
        advantages = advantages.reshape(-1, 1)

        for indices in sampler:
            # obs size [T+1 N M Dim]-->[T N M Dim]-->[T*N*M,Dim]-->[index,Dim]
            if self._mixed_obs:
                share_obs_batch = {}
                obs_batch = {}
                for key in share_obs.keys():
                    share_obs_batch[key] = share_obs[key][indices]
                for key in obs.keys():
                    obs_batch[key] = obs[key][indices]
            else:
                share_obs_batch = share_obs[indices]
                obs_batch = obs[indices]

            actions_batch = actions[indices]
            if self.available_actions is not None:
                available_actions_batch = available_actions[indices]
            else:
                available_actions_batch = None
            value_preds_batch = value_preds[indices]
            return_batch = returns[indices]
            masks_batch = masks[indices]
            active_masks_batch = active_masks[indices]
            old_action_log_probs_batch = action_log_probs[indices]
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages[indices]

            yield share_obs_batch, obs_batch, None, None, actions_batch, value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, adv_targ, available_actions_batch

    def naive_recurrent_generator(self, advantages, num_mini_batch):
        num_steps, num_envs, num_agents = self.rewards.shape[0:3]
        batch_size = num_envs*num_agents
        assert num_envs*num_agents >= num_mini_batch, (
            "PPO requires the number of processes ({})* number of agents ({}) "
            "to be greater than or equal to the number of "
            "PPO mini batches ({}).".format(num_envs, num_agents, num_mini_batch))
        num_envs_per_batch = batch_size // num_mini_batch
        perm = torch.randperm(batch_size).numpy()

        share_obs = self.share_obs.reshape(-1, batch_size, *self.share_obs.shape[3:])
        obs = self.obs.reshape(-1, batch_size, *self.obs.shape[3:])

        actions = self.actions.reshape(-1, batch_size, self.actions.shape[-1])
        if self.available_actions is not None:
            available_actions = self.available_actions.reshape(
                -1, batch_size, self.available_actions.shape[-1])
        value_preds = self.value_preds.reshape(-1, batch_size, 1)
        returns = self.returns.reshape(-1, batch_size, 1)
        masks = self.masks.reshape(-1, batch_size, 1)
        active_masks = self.active_masks.reshape(-1, batch_size, 1)
        action_log_probs = self.action_log_probs.reshape(
            -1, batch_size, self.action_log_probs.shape[-1])
        advantages = advantages.reshape(-1, batch_size, 1)

        for start_ind in range(0, batch_size, num_envs_per_batch):

            ind = perm[start_ind: start_ind + num_envs_per_batch]
            share_obs_batch = share_obs[:-1, ind]
            obs_batch = obs[:-1, ind]
            rnn_states_batch = self.rnn_states[0, :, ind]
            rnn_states_critic_batch = self.rnn_states_critic[0, :, ind]
            actions_batch = actions[:, ind]
            
            value_preds_batch = value_preds[:-1, ind]
            return_batch = returns[:-1, ind]
            masks_batch = masks[:-1, ind]
            active_masks_batch = active_masks[:-1, ind]
            old_action_log_probs_batch = action_log_probs[:, ind]
            adv_targ = advantages[:, ind]

            # Flatten the (T, N, ...) from_numpys to (T * N, ...)

            assert rnn_states_batch.dim()==3
            yield share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, adv_targ, None

    def recurrent_generator(self, advantages: torch.Tensor, num_mini_batches, chunk_length):
        T,  num_envs, num_agents = self.rewards.shape[0:3]
        B = num_envs * num_agents
        assert T % chunk_length == 0
        T_ = T//chunk_length
        B_ = T_ * B
        assert B_ % num_mini_batches == 0

        """
        (T, num_envs, num_agents, *)
        -> (T, B, *)
        -> (T//chunk_length, chunk_len, B, *)
        -> (chunk_len, T//chunk_len, B, *)
        -> (chunk_len, T//chunk_len * B, *)
        so that 
        when indices is a `LongTensor` of length minibatch_size=T//chunk_len * B / num_minibatch,
        tensor[:, indices] is of shape (chunk_len, minibatch_size, *)
        """
        share_obs = self.share_obs[:-1].reshape(T_, chunk_length, B, -1).transpose(0, 1).reshape(chunk_length, B_, -1)
        obs = self.obs[:-1].reshape(T_, chunk_length, B, -1).transpose(0, 1).reshape(chunk_length, B_, -1)
        actions = self.actions.reshape(T_, chunk_length, B, -1).transpose(0, 1).reshape(chunk_length, B_, -1)
        action_log_probs = self.action_log_probs.reshape(T_, chunk_length, B, -1).transpose(0, 1).reshape(chunk_length, B_, -1)
        value_preds = self.value_preds[:-1].reshape(T_, chunk_length, B, -1).transpose(0, 1).reshape(chunk_length, B_, -1)
        returns = self.returns[:-1].reshape(T_, chunk_length, B, -1).transpose(0, 1).reshape(chunk_length, B_, -1)
        masks = self.masks[:-1].reshape(T_, chunk_length, B, -1).transpose(0, 1).reshape(chunk_length, B_, -1)

        active_masks = self.active_masks[:-1].reshape(T_, chunk_length, B, -1).transpose(0, 1).reshape(chunk_length, B_, -1)
        if self.available_actions is not None:
            available_actions = self.available_actions[:-1].reshape(T_, chunk_length, B, -1).transpose(0, 1).reshape(chunk_length, B_, -1)
        advantages = advantages.reshape(T_, chunk_length, B, -1).transpose(0, 1).reshape(chunk_length, B_, -1)

        # (T, num_layers, B, H) -> (T//chunk_len, chunk_len, num_layers, B, H) -> (chunk_len, num_layers, T//chunk_len * B, H)
        rnn_states = self.rnn_states[:-1].reshape(T_, chunk_length, *self.rnn_states.shape[1:])\
            .permute(1, 2, 0, 3, 4).reshape(chunk_length, self.recurrent_N, B_, self.hidden_size)
        rnn_states_critic = self.rnn_states_critic[:-1].reshape(T_, chunk_length, *self.rnn_states_critic.shape[1:])\
            .permute(1, 2, 0, 3, 4).reshape(chunk_length, self.recurrent_N, B_, self.hidden_size)

        perm = torch.randperm(B_).reshape(num_mini_batches, -1)

        for indices in perm:
            share_obs_batch = share_obs[:, indices]
            obs_batch = obs[:, indices]
            actions_batch = actions[:, indices]
            action_log_probs_batch = action_log_probs[:, indices]
            value_preds_batch = value_preds[:, indices]
            returns_batch = returns[:, indices]
            masks_batch = masks[:, indices]
            active_masks_batch = active_masks[:, indices]
            adv_targ = advantages[:, indices]
            if self.available_actions is not None:
                available_actions_batch = available_actions[:, indices]
            else:
                available_actions_batch = None
            rnn_states_batch = rnn_states[0, :, indices]
            rnn_states_critic_batch = rnn_states_critic[0, :, indices]
            yield share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, value_preds_batch, returns_batch, masks_batch, active_masks_batch, action_log_probs_batch, adv_targ, available_actions_batch

