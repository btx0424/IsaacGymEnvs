    
from dataclasses import dataclass
import time
from typing import Dict
from isaacgymenvs.learning.mappo.algorithms.rmappo import MAPPOPolicy
from isaacgymenvs.tasks.base.vec_task import MultiAgentVecTask
import wandb
import os
import numpy as np
from itertools import chain
import torch
from tensorboardX import SummaryWriter
from isaacgymenvs.learning.mappo.utils.shared_buffer import SharedReplayBuffer

import socket
import psutil
import slackweb
from torchvision.utils import make_grid
webhook_url = " https://hooks.slack.com/services/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"
class Runner(object):
    def __init__(self, config):

        self.all_args = config['all_args']
        self.envs: MultiAgentVecTask = config['envs']
        # self.eval_envs = config['eval_envs']
        self.device = config['device']     

        # parameters
        self.use_centralized_V = self.all_args.use_centralized_V

        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.n_render_rollout_threads = self.all_args.n_render_rollout_threads
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        self.hidden_size = self.all_args.hidden_size

        self.use_single_network = self.all_args.use_single_network
        self.recurrent_N = self.all_args.recurrent_N

        # dir
        self.model_dir = self.all_args.model_dir

    def run(self):
        raise NotImplementedError

    def warmup(self):
        raise NotImplementedError

    def collect(self, step):
        raise NotImplementedError

    def insert(self, data):
        raise NotImplementedError
    
    @torch.no_grad()
    def compute(self):
        self.policy.prep_rollout()
        next_values = self.policy.get_values(
            self.buffer.share_obs[-1].flatten(end_dim=1),
            self.buffer.rnn_states_critic[-1].flatten(end_dim=1),
            self.buffer.masks[-1].flatten(end_dim=1))
        next_values = next_values.reshape(self.n_rollout_threads, self.num_agents, 1)
        self.buffer.compute_returns(next_values, self.policy.value_normalizer)
    
    def train(self):
        self.policy.prep_training()
        train_infos = self.policy.train(self.buffer)      
        self.buffer.after_update()
        self.log_system()
        return train_infos

    def save(self, num_steps):
        return
        if self.use_single_network:
            policy_model = self.trainer.policy.model
            torch.save(policy_model.state_dict(), str(self.save_dir) + f"/model_{num_steps}.pt")
        else:
            policy_actor = self.trainer.policy.actor
            torch.save(policy_actor.state_dict(), str(self.save_dir) + f"/actor_{num_steps}.pt")
            policy_critic = self.trainer.policy.critic
            torch.save(policy_critic.state_dict(), str(self.save_dir) + f"/critic_{num_steps}.pt")

    def restore(self):
        if self.use_single_network:
            policy_model_state_dict = torch.load(str(self.model_dir) + '/model.pt', map_location=self.device)
            self.policy.model.load_state_dict(policy_model_state_dict)
        else:
            policy_actor_state_dict = torch.load(str(self.model_dir) + '/actor.pt', map_location=self.device)
            self.policy.actor.load_state_dict(policy_actor_state_dict)
            if not (self.all_args.use_render or self.all_args.use_eval):
                policy_critic_state_dict = torch.load(str(self.model_dir) + '/critic.pt', map_location=self.device)
                self.policy.critic.load_state_dict(policy_critic_state_dict)
 
    def log_train(self, train_infos, total_num_steps):
        for k, v in train_infos.items():
            if self.use_wandb:
                wandb.log({k: v}, step=total_num_steps)
            else:
                self.writter.add_scalars(k, {k: v}, total_num_steps)

    def log_env(self, env_infos, total_num_steps):
        for k, v in env_infos.items():
            if len(v) > 0:
                if self.use_wandb:
                    wandb.log({k: np.mean(v)}, step=total_num_steps)
                else:
                    self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)
    
    def log(self, infos: Dict, steps: int):
        """
        Log training info.
        :param train_infos: (dict) information about training update.
        :param total_num_steps: (int) total number of training env steps.
        """
        if self.use_wandb:
            wandb.log(infos, step=steps)
        else:
            for k, v in infos.items():
                self.writter.add_scalar(k, v, steps)

    def log_gif(self, tag, vid_tensor, steps):
        """
        vid_tensor: (T, N, C, W, H)
        """
        vid_tensor = torch.from_numpy(vid_tensor)
        vid_tensor = torch.stack([make_grid(frame, nrow=5) for frame in  vid_tensor]).unsqueeze(0)
        if self.use_wandb:
            wandb.log({"gif": wandb.Video(vid_tensor, fps=8, format="gif")}, step=steps)
        else:
            self.writter.add_video(tag, vid_tensor, steps, fps=8)

    def log_system(self):
        # RRAM
        mem = psutil.virtual_memory()
        total_mem = float(mem.total) / 1024 / 1024 / 1024
        used_mem = float(mem.used) / 1024 / 1024 / 1024
        if used_mem/total_mem > 0.95:
            slack = slackweb.Slack(url=webhook_url)
            host_name = socket.gethostname()
            slack.notify(text="Host {}: occupied memory is *{:.2f}*%!".format(host_name, used_mem/total_mem*100))
