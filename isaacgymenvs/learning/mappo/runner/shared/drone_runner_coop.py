from cmath import inf
import time
import numpy as np
import torch
from .base_runner import Runner
from tqdm import tqdm

def _t2n(x):
    return x.detach().cpu().numpy()

def update_vel(phase_count, all_vel_curriculum, size):
    return np.ones(size) * all_vel_curriculum[phase_count]

class DroneRunner(Runner):
    def __init__(self, config):
        all_args = config["all_args"]
        self.eval_episodes = all_args.eval_episodes
        self.episodes_per_update = all_args.episodes_per_update

        super().__init__(config)
        print("obs space:", self.envs.observation_space[0])
        print("act space:", self.envs.action_space[0])

        # self.num_drones = self.envs.getattr_single("NUM_DRONES")
        # self.predators = self.envs.getattr_single("predators")
        # self.preys = self.envs.getattr_single("preys")
        # self.max_reward = self.envs.getattr_single("max_reward")
        # self.horizon = self.envs.getattr_single("max_steps")
        # self.cl_max_vel = self.all_args.max_vel
        # self.cl_min_vel = self.all_args.min_vel
        # self.success_threshold = self.all_args.success_threshold

    def run(self):
        self.warmup()

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        reward_normalizer = self.episodes_per_update * self.n_rollout_threads * 500
        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            episode_count = 0
            collision_penalty = 0
            action_stats = []
            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect(step)
                # Obser reward and next obs, (n_threads, n_agents, *)
                obs, rewards, dones, infos = self.envs.step(actions)
                action_stats.append(actions)
                data = obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic

                # insert data into buffer
                self.insert(data)

                # record statistics
                envs_done = torch.nonzero(torch.all(dones, axis=1))
                episode_count += len(envs_done)

            # compute return and update network
            self.compute()
            train_infos = self.train()
            
            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            
            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save(total_num_steps)

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print(
                    f"num_agents:{self.num_agents}, updates: {episode}/{episodes} iters, " \
                    + f"total num timesteps: {total_num_steps}/{self.num_env_steps}, FPS: {int(total_num_steps / (end - start))}, " \
                    + f"run time: {end - start}"
                )
                train_infos["reward"] = torch.mean(self.buffer.rewards) * self.episode_length
                print("reward:", train_infos["reward"])
                # self.log(train_infos, total_num_steps)

            # eval
            # if episode % self.eval_interval == 0 and self.use_eval:
            #     self.eval(total_num_steps)

    def warmup(self):
        # reset env
        obs = self.envs.reset()
        share_obs = obs

        self.buffer.share_obs[0] = share_obs
        self.buffer.obs[0] = obs

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        value, action, action_log_prob, rnn_states, rnn_states_critic \
            = self.trainer.policy.get_actions(
                self.buffer.share_obs[step].flatten(end_dim=1),
                self.buffer.obs[step].flatten(end_dim=1),
                self.buffer.rnn_states[step].flatten(end_dim=1),
                self.buffer.rnn_states_critic[step].flatten(end_dim=1),
                self.buffer.masks[step].flatten(end_dim=1))
        # [self.envs, agents, dim]
        values = value.reshape(self.n_rollout_threads, self.num_agents, -1)
        actions = action.reshape(self.n_rollout_threads, self.num_agents, -1)
        action_log_probs = action_log_prob.reshape(self.n_rollout_threads, self.num_agents, -1)
        rnn_states = rnn_states.reshape(self.n_rollout_threads, self.num_agents, *rnn_states.shape[1:])
        rnn_states_critic  = rnn_states_critic.reshape(self.n_rollout_threads, self.num_agents, *rnn_states_critic.shape[1:])

        # values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        # actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        # action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        # rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        # rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))

        return values, actions, action_log_probs, rnn_states, rnn_states_critic
    
    def insert(self, data):
        obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        rnn_states[dones == True] = 0
        rnn_states_critic[dones == True] = 0
        
        dones_env = torch.all(dones, axis=1)

        masks = torch.ones((self.n_rollout_threads, self.num_agents, 1))
        masks[dones == True] = 0

        active_masks = torch.ones((self.n_rollout_threads, self.num_agents, 1))
        active_masks[dones == True] = 0
        active_masks[dones_env == True] = 1

        rewards.unsqueeze_(-1)
        share_obs = obs

        self.buffer.insert(share_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs, values, rewards, masks, active_masks=active_masks)
    
    @torch.no_grad()
    def eval(self, total_num_steps):
        assert self.eval_episodes == self.n_eval_rollout_threads 
        
        eval_reward = 0

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, *self.buffer.rnn_states.shape[2:]), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        eval_episode_rewards = []
        eval_obs = self.eval_envs.reset()

        frames = []

        for eval_step in tqdm(range(self.eval_envs.getattr_single("max_steps"))):
            self.trainer.prep_rollout()
            eval_action, eval_rnn_states = self.trainer.policy.act(
                np.concatenate(eval_obs),
                np.concatenate(eval_rnn_states),
                np.concatenate(eval_masks),
                deterministic=False)

            eval_actions = np.array(np.split(_t2n(eval_action), self.n_eval_rollout_threads))

            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))

            # Observe reward and next obs
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions)

            eval_rewards = eval_rewards.reshape([-1, self.num_agents])
            eval_episode_rewards.append(eval_rewards)

            eval_rnn_states[eval_dones == True] = np.zeros(
                ((eval_dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)
            if np.all(eval_dones):
                break
            
            if eval_step % 4 ==0: 
                captures = self.eval_envs.render("mini_map") # (N, H, W, C)
                frames.append(captures.squeeze())

        # (T, N, H, W, C) -> (T, N, C, H, W)
        frames = np.stack(frames)
        frames = np.transpose(frames, (0, 1, 4, 2, 3)) 
        # (step, rollout, ) -> (rollout, )
        eval_episode_rewards = np.array(eval_episode_rewards)

        eval_env_infos = {
            "eval_episode_length": eval_step,
            "eval_predators_reward": eval_episode_rewards[:, :, :self.all_args.num_predators].sum() / self.eval_episodes / self.max_reward,
            "eval_preys_reward": eval_episode_rewards[:, :, self.all_args.num_predators:].sum() / self.eval_episodes / self.max_reward,
        }

        print(eval_env_infos)

        self.log(eval_env_infos, total_num_steps)
        self.log_gif("eval", frames, total_num_steps)

import gc
import datetime
import inspect

dtype_memory_size_dict = {
    torch.float64: 64/8,
    torch.double: 64/8,
    torch.float32: 32/8,
    torch.float: 32/8,
    torch.float16: 16/8,
    torch.half: 16/8,
    torch.int64: 64/8,
    torch.long: 64/8,
    torch.int32: 32/8,
    torch.int: 32/8,
    torch.int16: 16/8,
    torch.short: 16/6,
    torch.uint8: 8/8,
    torch.int8: 8/8,
}
# compatibility of torch1.0
if getattr(torch, "bfloat16", None) is not None:
    dtype_memory_size_dict[torch.bfloat16] = 16/8
if getattr(torch, "bool", None) is not None:
    dtype_memory_size_dict[torch.bool] = 8/8 # pytorch use 1 byte for a bool, see https://github.com/pytorch/pytorch/issues/41571

def get_mem_space(x):
    try:
        ret = dtype_memory_size_dict[x]
    except KeyError:
        print(f"dtype {x} is not supported!")
    return ret

class MemTracker(object):
    """
    Class used to track pytorch memory usage
    Arguments:
        detail(bool, default True): whether the function shows the detail gpu memory usage
        path(str): where to save log file
        verbose(bool, default False): whether show the trivial exception
        device(int): GPU number, default is 0
    """
    def __init__(self, detail=True, path='', verbose=False, device=0):
        self.print_detail = detail
        self.last_tensor_sizes = set()
        self.gpu_profile_fn = path + f'{datetime.datetime.now():%d-%b-%y-%H:%M:%S}-gpu_mem_track.txt'
        self.verbose = verbose
        self.begin = True
        self.device = device

    def get_tensors(self):
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                    tensor = obj
                else:
                    continue
                if tensor.is_cuda:
                    yield tensor
            except Exception as e:
                if self.verbose:
                    print('A trivial exception occured: {}'.format(e))

    def get_tensor_usage(self):
        sizes = [np.prod(np.array(tensor.size())) * get_mem_space(tensor.dtype) for tensor in self.get_tensors()]
        return np.sum(sizes) / 1024**2

    def get_allocate_usage(self):
        return torch.cuda.memory_allocated() / 1024**2

    def clear_cache(self):
        gc.collect()
        torch.cuda.empty_cache()

    def print_all_gpu_tensor(self, file=None):
        for x in self.get_tensors():
            print(x.size(), x.dtype, np.prod(np.array(x.size()))*get_mem_space(x.dtype)/1024**2, file=file)

    def track(self):
        """
        Track the GPU memory usage
        """
        frameinfo = inspect.stack()[1]
        where_str = frameinfo.filename + ' line ' + str(frameinfo.lineno) + ': ' + frameinfo.function

        with open(self.gpu_profile_fn, 'a+') as f:

            if self.begin:
                f.write(f"GPU Memory Track | {datetime.datetime.now():%d-%b-%y-%H:%M:%S} |"
                        f" Total Tensor Used Memory:{self.get_tensor_usage():<7.1f}Mb"
                        f" Total Allocated Memory:{self.get_allocate_usage():<7.1f}Mb\n\n")
                self.begin = False

            if self.print_detail is True:
                ts_list = [(tensor.size(), tensor.dtype) for tensor in self.get_tensors()]
                new_tensor_sizes = {(type(x),
                                    tuple(x.size()),
                                    ts_list.count((x.size(), x.dtype)),
                                    np.prod(np.array(x.size()))*get_mem_space(x.dtype)/1024**2,
                                    x.dtype) for x in self.get_tensors()}
                for t, s, n, m, data_type in new_tensor_sizes - self.last_tensor_sizes:
                    f.write(f'+ | {str(n)} * Size:{str(s):<20} | Memory: {str(m*n)[:6]} M | {str(t):<20} | {data_type}\n')
                for t, s, n, m, data_type in self.last_tensor_sizes - new_tensor_sizes:
                    f.write(f'- | {str(n)} * Size:{str(s):<20} | Memory: {str(m*n)[:6]} M | {str(t):<20} | {data_type}\n')

                self.last_tensor_sizes = new_tensor_sizes

            f.write(f"\nAt {where_str:<50}"
                    f" Total Tensor Used Memory:{self.get_tensor_usage():<7.1f}Mb"
                    f" Total Allocated Memory:{self.get_allocate_usage():<7.1f}Mb\n\n")