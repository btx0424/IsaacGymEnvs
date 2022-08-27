import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from isaacgymenvs.learning.mappo.algorithms.utils.util import init, check
from isaacgymenvs.learning.mappo.algorithms.utils.cnn import CNNBase
from isaacgymenvs.learning.mappo.algorithms.utils.mlp import MLPBase, MLPLayer
from isaacgymenvs.learning.mappo.algorithms.utils.mix import MIXBase
from isaacgymenvs.learning.mappo.algorithms.utils.rnn import RNNLayer
from isaacgymenvs.learning.mappo.algorithms.utils.act import ACTLayer
from isaacgymenvs.learning.mappo.algorithms.utils.popart import PopArt
from isaacgymenvs.learning.mappo.utils.util import get_shape_from_obs_space

class R_Actor(nn.Module):
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super(R_Actor, self).__init__()
        self.hidden_size = args.hidden_size

        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal 
        self._activation_id = args.activation_id
        self._use_policy_active_masks = args.use_policy_active_masks 
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_influence_policy = args.use_influence_policy
        self._influence_layer_N = args.influence_layer_N 
        self._use_policy_vhead = args.use_policy_vhead
        self._use_popart = args.use_popart 
        self._recurrent_N = args.recurrent_N 
        self.tpdv = dict(dtype=torch.float32, device=device)

        obs_shape = get_shape_from_obs_space(obs_space)

        self.base = MLPBase(args, obs_shape, use_attn_internal=args.use_attn_internal, use_cat_self=True)
        
        input_size = self.base.output_size

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(input_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)
            input_size = self.hidden_size

        self.act = ACTLayer(action_space, input_size, self._use_orthogonal, self._gain, sde=args.sde)

        self.to(device)

    def forward(self, 
            obs, 
            rnn_states=None, 
            masks=None, 
            available_actions=None, deterministic=False):

        actor_features = self.base(obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        actions, action_log_probs = self.act(actor_features, available_actions, deterministic)
        
        return actions, action_log_probs, rnn_states

    def evaluate_actions(self, obs, rnn_states, action, masks, available_actions=None, active_masks=None):
        
        actor_features = self.base(obs)
        
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        action_log_probs, dist_entropy = self.act.evaluate_actions(actor_features, action, available_actions, active_masks = active_masks if self._use_policy_active_masks else None)
       
        return action_log_probs, dist_entropy

    def get_feature(self, obs, rnn_state=None, mask=None):
        features = self.base(obs)
        if self._use_recurrent_policy:
            features, _ = self.rnn(features, rnn_state, mask)
        return features

class R_Critic(nn.Module):
    def __init__(self, args, share_obs_space, device=torch.device("cpu")):
        super(R_Critic, self).__init__()
        self.hidden_size = args.hidden_size
        self._use_orthogonal = args.use_orthogonal  
        self._activation_id = args.activation_id     
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_influence_policy = args.use_influence_policy
        self._use_popart = args.use_popart
        self._influence_layer_N = args.influence_layer_N
        self._recurrent_N = args.recurrent_N
        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]

        share_obs_shape = get_shape_from_obs_space(share_obs_space)

        self.base = MLPBase(args, share_obs_shape, use_attn_internal=args.use_attn_internal, use_cat_self=True)

        input_size = self.base.output_size

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(input_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)
            input_size = self.hidden_size

        if self._use_influence_policy:
            self.mlp = MLPLayer(share_obs_shape[0], self.hidden_size,
                              self._influence_layer_N, self._use_orthogonal, self._activation_id)
            input_size += self.hidden_size

        def init_(m): 
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        if self._use_popart:
            self.v_out = init_(PopArt(input_size, 1, device=device))
        else:
            self.v_out = init_(nn.Linear(input_size, 1))

        self.to(device)

    def forward(self, share_obs, rnn_states, masks):

        critic_features = self.base(share_obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)

        values = self.v_out(critic_features)

        return values, rnn_states

    def get_feature_and_value(self, state, rnn_state=None, mask=None) -> torch.Tensor:
        critic_features = self.base(state)
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            critic_features, rnn_states = self.rnn(critic_features, rnn_states, mask)
        values = self.v_out(critic_features)
        return critic_features, values
    
    def get_value(self, state, rnn_state=None, mask=None) -> torch.Tensor:
        critic_features = self.base(state)
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            critic_features, rnn_states = self.rnn(critic_features, rnn_states, mask)
        values = self.v_out(critic_features)
        return values
