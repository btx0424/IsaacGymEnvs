import numpy as np
import os
import random
import pdb
import wandb
from pathlib import Path
from scipy.spatial.distance import cdist
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

def Hash(buffer):
    _, state_dim = np.array(buffer).shape
    return np.sum(np.array(buffer) * np.logspace(start=0, stop=state_dim - 1, base=2, num=state_dim), axis=1)

class goal_proposal():
    def __init__(self, config, env_name, scenario_name, buffer_capacity, proposal_batch, device, horizon):
        # TODO: zip env parameters
        self.config = config
        self.env_name = env_name
        self.scenario_name = scenario_name
        # self.alpha = config.alpha
        self.buffer_capacity = buffer_capacity
        self.proposal_batch = proposal_batch
        # active: restart_p, easy: restart_easy, unif: 1-restart_p-restart_easy
        self.restart_p = config.restart_p # means medium cases 
        self.restart_easy = config.restart_easy # easy cases
        self.medium_buffer = [] # store restart states
        self.medium_buffer_dist = [] # for diversified buffer
        self.medium_buffer_score = []
        self.easy_buffer = [] # collect easy cases (high return)
        self.easy_buffer_dist = []
        self.easy_buffer_score = []
        self.device = device
        self.horizon = horizon
        self.use_diversified = config.use_diversified
        self.sigma_min = config.sigma_min # < min = hard cases
        self.sigma_max = config.sigma_max # > max = easy cases

    def init_env_config(self, env_config):
        self.num_agents = env_config['num_agents']

    def restart_sampling_withEasy(self):
        starts = []
        restart_index = []

        # sample from medium and easy
        num_medium = 0
        num_easy = 0
        for index in range(self.proposal_batch):
            # 0:unif, 1:medium, 2:easy
            choose_flag = np.random.choice([0,1,2],size=1,replace=True,
                                p=[1 - self.restart_p - self.restart_easy, self.restart_p, self.restart_easy])[0]
            if choose_flag == 1:
                num_medium += 1
            elif choose_flag == 2:
                num_easy += 1
        if num_medium > len(self.medium_buffer): num_medium = len(self.medium_buffer)
        if num_easy > len(self.easy_buffer): num_easy = len(self.easy_buffer)

        new_starts, restart_index = self.uniform_from_buffer(buffer=self.medium_buffer,starts_length=num_medium)
        starts += new_starts
        easy_starts, _ = self.uniform_from_buffer(buffer=self.easy_buffer,starts_length=num_easy)
        starts += easy_starts
        starts += [None] * (self.proposal_batch - num_medium - num_easy)
        
        return starts, restart_index

    # direct add states
    def add_states(self, states, scores):
        thread = self.config.n_rollout_threads
        num_batch = int(self.config.episode_length / self.horizon)

        add_states = []
        add_scores = []
        if self.config.use_start_states:
            add_index_begin = np.linspace(start=0, stop=len(states) - thread * self.horizon, num=num_batch, dtype=int)
            for add_index in add_index_begin:
                add_states += states[add_index : add_index + thread].copy()
                add_scores += scores[add_index : add_index + thread].copy()
        else:
            add_states = states.copy()
            add_scores = scores.copy()

        start = time.time()
        if self.config.use_states_clip:
            if self.env_name == 'drone' and self.scenario_name == 'Occupation_empty':               
                # delete illegal states
                for state_id in reversed(range(len(add_states))):
                    if self.illegal_task(add_states[state_id]):
                        del add_states[state_id]
                        del add_scores[state_id]
        end = time.time()
        print('clip', end-start)

        # add states to buffer
        for state, score in zip(add_states, add_scores):
            if score >= self.sigma_max:
                self.easy_buffer += [state]
                self.easy_buffer_score += [score]
            elif score < self.sigma_max and score >= self.sigma_min:
                self.medium_buffer += [state]
                self.medium_buffer_score += [score]

        # update dist, priority
        if len(self.medium_buffer) > self.buffer_capacity:
            self.medium_buffer_dist = self.get_dist(self.medium_buffer, self.device)
        if len(self.easy_buffer) > self.buffer_capacity:
            self.easy_buffer_dist = self.get_dist(self.easy_buffer, self.device)

        # delete states by novelty
        start = time.time()
        if len(self.medium_buffer) > self.buffer_capacity:
            self.medium_buffer_dist, self.medium_buffer, self.medium_buffer_score = self.buffer_sort(self.medium_buffer_dist, self.medium_buffer, self.medium_buffer_score)
            # self.medium_buffer_dist, self.medium_buffer = self.buffer_sort(self.medium_buffer_dist, self.medium_buffer)
            self.medium_buffer = self.medium_buffer[len(self.medium_buffer)-self.buffer_capacity:]
            self.medium_buffer_dist = self.medium_buffer_dist[len(self.medium_buffer_dist)-self.buffer_capacity:]
            self.medium_buffer_score = self.medium_buffer_score[len(self.medium_buffer_score)-self.buffer_capacity:]
        end = time.time()
        print('delete medium', end-start)

        start = time.time()
        if len(self.easy_buffer) > self.buffer_capacity:
            self.easy_buffer_dist, self.easy_buffer, self.easy_buffer_score = self.buffer_sort(self.easy_buffer_dist, self.easy_buffer, self.easy_buffer_score)
            # self.easy_buffer_dist, self.easy_buffer = self.buffer_sort(self.easy_buffer_dist, self.easy_buffer)
            self.easy_buffer = self.easy_buffer[len(self.easy_buffer)-self.buffer_capacity:]
            self.easy_buffer_dist = self.easy_buffer_dist[len(self.easy_buffer_dist)-self.buffer_capacity:]
            self.easy_buffer_score = self.easy_buffer_score[len(self.easy_buffer_score)-self.buffer_capacity:]
        end = time.time()
        print('delete easy', end-start)

        self.medium_buffer = [np.array(state) for state in self.medium_buffer]
        self.easy_buffer = [np.array(state) for state in self.easy_buffer]

    def uniform_from_buffer(self, buffer, starts_length):
        self.choose_index = np.random.choice(range(len(buffer)),size=starts_length,replace=True)
        starts = []
        for index in self.choose_index:
            starts.append(buffer[index])
        return starts, self.choose_index

    def priority_sampling(self, starts_length):
        self.buffer_p = []
        sum_p = 0
        for priority in self.buffer_priority:
            sum_p += priority**self.alpha
        for priority in self.buffer_priority:
            self.buffer_p.append(priority**self.alpha / sum_p)
        self.choose_index = np.random.choice(range(len(self.buffer)),size=starts_length,replace=True,p=self.buffer_p)
        starts = []
        for index in self.choose_index:
            starts.append(self.buffer[index])
        return starts, self.choose_index

    def priority_sampling_withdelete(self, starts_length):
        self.buffer_p = []
        sum_p = 0
        for priority in self.buffer_priority:
            sum_p += priority**self.alpha
        for priority in self.buffer_priority:
            self.buffer_p.append(priority**self.alpha / sum_p)
        self.choose_index = np.random.choice(range(len(self.buffer)),size=starts_length,replace=True,p=self.buffer_p)
        starts = []
        for index in self.choose_index:
            starts.append(self.buffer[index])
        # delete states
        self.choose_index = np.sort(np.unique(self.choose_index))
        for index in reversed(self.choose_index):
            del self.buffer[index]
            del self.buffer_priority[index]
        self.choose_index = np.flipud(self.choose_index)
        return starts, self.choose_index

    def priority_sampling_woreplacement(self, starts_length):
        self.buffer_p = []
        sum_p = 0
        for priority in self.buffer_priority:
            sum_p += priority**self.alpha
        for priority in self.buffer_priority:
            self.buffer_p.append(priority**self.alpha / sum_p)
        self.choose_index = np.random.choice(range(len(self.buffer)),size=starts_length,replace=False,p=self.buffer_p)
        self.choose_index = np.sort(self.choose_index)
        starts = []
        for index in reversed(self.choose_index):
            starts.append(self.buffer[index])
            del self.buffer[index]
            del self.buffer_priority[index]
        self.choose_index = np.flipud(self.choose_index)
        return starts, self.choose_index

    def random_sampling_woreplacement(self, starts_length):
        self.buffer_p = []
        # for priority in self.buffer_priority:
        #     sum_p += priority**self.alpha
        # for priority in self.buffer_priority:
        #     self.buffer_p.append(priority**self.alpha / sum_p)
        self.buffer_p = [1 / len(self.buffer)] * len(self.buffer)
        self.choose_index = np.random.choice(range(len(self.buffer)),size=starts_length,replace=False,p=self.buffer_p)
        self.choose_index = np.sort(self.choose_index)
        starts = []
        for index in reversed(self.choose_index):
            starts.append(self.buffer[index])
            del self.buffer[index]
            del self.buffer_priority[index]
        return starts, self.choose_index

    def rank_sampling(self, buffer, starts_length):
        self.choose_index = [i + len(buffer) - starts_length for i in range(starts_length)]
        starts = self.buffer[(len(buffer) - starts_length):]
        return starts, self.choose_index

    def get_dist_and_update_priority(self, buffer, buffer_priority, device):
        # topk=5
        # dist = cdist(np.array(buffer).reshape(len(buffer),-1),np.array(buffer).reshape(len(buffer),-1),metric='euclidean')
        # if len(buffer) < topk+1:
        #     dist_k = dist
        #     novelty = np.sum(dist_k,axis=1)/len(buffer)
        # else:
        #     dist_k = np.partition(dist,topk,axis=1)[:,0:topk]
        #     novelty = np.sum(dist_k,axis=1)/topk

        n = len(buffer)
        buffer_array = torch.from_numpy(np.array(buffer)).float().to(device)
        topk = 5
        if n // 500 > 5:
            chunk = n // 500
            dist = []
            priority = []
            for i in range((n // chunk) + 1):
                b = buffer_array[i * chunk : (i+1) * chunk]
                d = self._euclidean_dist(b, buffer_array)
                # d = torch.matmul(b, buffer_array.transpose(0,1))
                dist_nearest_chunk, dist_chunk_index = torch.topk(d, k=topk, dim=1, largest=False)
                dist_nearest_chunk = dist_nearest_chunk.cpu().numpy()
                if self.use_Guassian_smoothing or self.use_Guassian_diversified:
                    # delete self dist and index
                    dist_nearest_chunk_other = dist_nearest_chunk[:,1:]
                    dist_chunk_index_other = dist_chunk_index[:,1:]
                    dist_weight = np.exp(-dist_nearest_chunk_other) / np.sum(np.exp(-dist_nearest_chunk_other),axis=1).reshape(-1,1)
                    nearest_buffer_priority = np.array(buffer_priority)[dist_chunk_index_other.cpu().numpy()]
                    priority_chunk = np.sum(dist_weight * nearest_buffer_priority, axis=1)
                    priority.append(priority_chunk.copy())
                else: # box kernel
                    priority.append(np.mean(np.array(buffer_priority)[dist_chunk_index.cpu().numpy()],axis=1))
                if self.use_Guassian_diversified:
                    dist.append(np.sum(dist_weight * dist_nearest_chunk_other, axis=1))
                else:
                    dist.append(np.mean(dist_nearest_chunk,axis=1))
            dist = np.concatenate(dist, axis=0)
            priority = np.concatenate(priority, axis=0)
        else:
            d = self._euclidean_dist(buffer_array, buffer_array)
            dist_nearest, dist_index = torch.topk(d, k=topk, dim=1, largest=False)
            dist_nearest = dist_nearest.cpu().numpy()
            if self.use_Guassian_smoothing or self.use_Guassian_diversified:
                # delete self dist and index
                dist_nearest_other = dist_nearest[:,1:]
                dist_index_other = dist_index[:,1:]
                dist_weight = np.exp(-dist_nearest_other) / np.sum(np.exp(-dist_nearest_other),axis=1).reshape(-1,1)
                nearest_buffer_priority = np.array(buffer_priority)[dist_index_other.cpu().numpy()]
                priority = np.sum(dist_weight * nearest_buffer_priority, axis=1)
            else:
                priority = np.mean(np.array(buffer_priority)[dist_index.cpu().numpy()],axis=1)
            if self.use_Guassian_diversified:
                dist = np.sum(dist_weight * dist_nearest_other, axis=1)
            else:
                dist = np.mean(dist_nearest,axis=1)
        if self.use_smooth_weight:
            self.buffer_priority = priority.tolist().copy()
        return dist

    def get_dist(self, buffer, device):
        n = len(buffer)
        buffer_array = torch.from_numpy(np.array(buffer)).float().to(device)
        topk = 5
        if n // 500 > 5:
            chunk = n // 500
            dist = []
            for i in range((n // chunk) + 1):
                b = buffer_array[i * chunk : (i+1) * chunk]
                d = self._euclidean_dist(b, buffer_array)
                # d = torch.matmul(b, buffer_array.transpose(0,1))
                dist_nearest_chunk, dist_chunk_index = torch.topk(d, k=topk, dim=1, largest=False)
                dist_nearest_chunk = dist_nearest_chunk.cpu().numpy()
                dist.append(np.mean(dist_nearest_chunk,axis=1))
            dist = np.concatenate(dist, axis=0)
        else:
            d = self._euclidean_dist(buffer_array, buffer_array)
            dist_nearest, dist_index = torch.topk(d, k=topk, dim=1, largest=False)
            dist_nearest = dist_nearest.cpu().numpy()
            dist = np.mean(dist_nearest,axis=1)
        return dist

    def _euclidean_dist(self, x, y):
        """
        Args:
          x: pytorch Variable, with shape [m, d]
          y: pytorch Variable, with shape [n, d]
        Returns:
          dist: pytorch Variable, with shape [m, n]
        """
 
        m, n = x.size(0), y.size(0)
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
        yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy
        # dist - 2 * x * yT 
        dist.addmm_(1, -2, x, y.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        return dist

    def buffer_sort(self, list1, *args): # sort by list1, ascending order
        zipped = zip(list1,*args)
        sort_zipped = sorted(zipped,key=lambda x:(x[0],np.mean(x[1])))
        result = zip(*sort_zipped)
        return [list(x) for x in result]

    def easy_sampling(self, starts_length):
        if self.env_name == 'HideAndSeek' and self.scenario_name == 'quadrant':
            archive = []
            # for j in range(starts_length // 2): # ramp beside the door, to train use ramp
            #     ramp_xpos = np.random.randint(16,29)
            #     door_pos = np.random.randint(16,28)
            #     archive.append(np.array([25,3,25,10,3,10,25,7,0,ramp_xpos,16,0, door_pos, 15, door_pos + 1,15]))

            # for j in range(starts_length - starts_length // 2): # ramp far from the door, to train use ramp
            for j in range(starts_length): # ramp far from the door, to train use ramp
                # ramp_ypos = np.random.randint(2,14)
                door_pos = np.random.randint(1,14)
                delta = np.random.randint(-1,1)
                ramp_ypos = door_pos + delta
                ramp_ypos = np.clip(ramp_ypos, 2, 14)
                archive.append(np.array([25,3,25,10,3,10,25,7,0,13,ramp_ypos,0, 15, door_pos, 15,door_pos + 1]))
            return archive

    def fixed_sampling(self, starts_length):
        if self.env_name == 'HideAndSeek' and self.scenario_name == 'quadrant':
            archive = []
            # box at door and one ramp
            for j in range(starts_length): # ramp beside the door, to train use ramp
                archive.append(np.array([25,3,25,10,3,10,3,15,25,7,0,22,16,0, 20, 15, 21,15]))
            return archive

            for j in range(starts_length): # ramp far from the door, to train use ramp
                archive.append(np.array([25,3,25,10,3,10,3,15,25,7,0,13,3,0, 20, 15, 21,15]))
            return archive

    def load_node(self, mode_path, load_episode, role_id=None):
        if role_id is None:
            data_dir = mode_path + '/starts' + '/starts_' + str(load_episode)
            value_dir = mode_path + '/starts' + '/values_' + str(load_episode)
        else:
            data_dir = mode_path + '/starts/' + role_id + '/starts_' + str(load_episode)
            difficulty_data_dir = mode_path + '/starts/' + role_id + '/easy_starts_' + str(load_episode)
            value_dir = mode_path + '/starts/' + role_id + '/values_' + str(load_episode)

        # load task
        with open(data_dir,'r') as fp:
            data = fp.readlines()
        for i in range(len(data)):
            data[i] = np.array(data[i][1:-2].split(),dtype=int)
        data_true = []
        for i in range(len(data)):
            if data[i].shape[0]>5:
                data_true.append(data[i])

        # load difficulty task
        with open(difficulty_data_dir,'r') as fp:
            difficulty_data = fp.readlines()
        for i in range(len(difficulty_data)):
            difficulty_data[i] = np.array(difficulty_data[i][1:-2].split(),dtype=int)
        difficulty_data_true = []
        for i in range(len(difficulty_data)):
            if difficulty_data[i].shape[0]>5:
                difficulty_data_true.append(difficulty_data[i])

        # load value
        with open(value_dir,'r') as fp:
            values = fp.readlines()
        for i in range(len(values)):
            values[i] = np.array(values[i][1:-2].split(),dtype=float)
        
        self.buffer = copy.deepcopy(data_true)
        self.difficulty_buffer = copy.deepcopy(difficulty_data_true)
        self.buffer_priority = copy.deepcopy(np.array(values).reshape(-1).tolist())

    def save_node(self, dir_path, episode):
        # dir_path: '/home/chenjy/mappo-curriculum/' + args.model_dir
        save_path = Path(dir_path) / 'starts'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if len(self.medium_buffer) > 0:
            with open(save_path / ('medium_starts_%i' %(episode)),'w+') as fp:
                for line in self.medium_buffer:
                    fp.write(str(np.array(line).reshape(-1))+'\n')
            with open(save_path / ('medium_values_%i' %(episode)),'w+') as fp:
                for line in self.medium_buffer_score:
                    fp.write(str(np.array(line).reshape(-1))+'\n')
            with open(save_path / ('medium_dist_%i' %(episode)),'w+') as fp:
                for line in self.medium_buffer_dist:
                    fp.write(str(np.array(line).reshape(-1))+'\n')

        if len(self.easy_buffer) > 0:
            with open(save_path / ('easy_starts_%i' %(episode)),'w+') as fp:
                for line in self.easy_buffer:
                    fp.write(str(np.array(line).reshape(-1))+'\n')
            with open(save_path / ('easy_values_%i' %(episode)),'w+') as fp:
                for line in self.easy_buffer_score:
                    fp.write(str(np.array(line).reshape(-1))+'\n')
            with open(save_path / ('easy_dist_%i' %(episode)),'w+') as fp:
                for line in self.easy_buffer_dist:
                    fp.write(str(np.array(line).reshape(-1))+'\n')


    def illegal_task(self, task):
        if self.env_name == 'HideAndSeek' and self.scenario_name == 'quadrant':
            def in_quadrant(pos, obj_size):
                if pos[0] >= self.grid_size // 2 and pos[0] <= self.grid_size - obj_size - 1:
                    if pos[1] >= 1 and pos[1] <= self.grid_size // 2 - obj_size - 1:
                        return True
                return False
            
            def outside_quadrant(pos, obj_size):
                if pos[0] >= 1 and pos[0] <= self.grid_size // 2 - obj_size - 1:
                    if pos[1] >= 1 and pos[1] <= self.grid_size // 2 - obj_size - 1:
                        return True
                    elif pos[1] >= self.grid_size // 2 and pos[1] <= self.grid_size - obj_size - 1:
                        return True
                elif pos[0] >= self.grid_size // 2 and pos[0] <= self.grid_size - obj_size - 1:
                    if pos[1] >= self.grid_size // 2 and pos[1] <= self.grid_size - obj_size - 1:
                        return True
                return False

            def in_map(pos, obj_size):
                if pos[0] >= 1 and pos[0] <= self.grid_size - obj_size - 1:
                    if pos[1] >= 1 and pos[1] <= self.grid_size - obj_size - 1:
                        return True
                return False

            hider_pos = task[:self.num_hiders * 2]
            for hider_id in range(self.num_hiders):
                if self.quadrant_game_hider_uniform_placement:
                    if in_map(hider_pos[hider_id * 2 : (hider_id + 1) * 2], self.agent_size):
                        continue
                    else:
                        return True
                else:
                    if in_quadrant(hider_pos[hider_id * 2 : (hider_id + 1) * 2], self.agent_size):
                        continue
                    else:
                        return True

            seeker_pos = task[self.num_hiders * 2: self.num_hiders * 2 + self.num_seekers * 2]
            for seeker_id in range(self.num_seekers):
                if outside_quadrant(seeker_pos[seeker_id * 2 : (seeker_id + 1) * 2], self.agent_size):
                    continue
                else:
                    return True

            box_pos = task[(self.num_hiders + self.num_seekers) * 2 : (self.num_hiders + self.num_seekers) * 2 + self.num_boxes * 2]
            for box_id in range(self.num_boxes):
                if in_quadrant(box_pos[box_id * 2 : (box_id + 1) * 2], self.box_size):
                    continue
                else:
                    return True
            ramp_pos = task[(self.num_hiders + self.num_seekers) * 2 + self.num_boxes * 3 : (self.num_hiders + self.num_seekers) * 2 + self.num_boxes * 3 + self.num_ramps * 2]
            for ramp_id in range(self.num_ramps):
                if outside_quadrant(ramp_pos[ramp_id * 2 : (ramp_id + 1) * 2], self.ramp_size):
                    continue
                else:
                    return True
            return False

    def overlap_task(self, task):
        grid_map = np.zeros(shape=(self.grid_size,self.grid_size))
        
        hider_poses = task[:self.num_hiders * 2]
        for hider_id in range(self.num_hiders):
            hider_pos = hider_poses[hider_id * 2 : (hider_id + 1) * 2]
            if grid_map[hider_pos[0],hider_pos[1]] == 0:
                grid_map[hider_pos[0],hider_pos[1]] = 1
                continue
            else:
                return True

        seeker_poses = task[self.num_hiders * 2: self.num_hiders * 2 + self.num_seekers * 2]
        for seeker_id in range(self.num_seekers):
            seeker_pos = seeker_poses[seeker_id * 2 : (seeker_id + 1) * 2]
            if grid_map[seeker_pos[0],seeker_pos[1]] == 0:
                grid_map[seeker_pos[0],seeker_pos[1]] = 1
                continue
            else:
                return True

        box_poses = task[(self.num_hiders + self.num_seekers) * 2 : (self.num_hiders + self.num_seekers) * 2 + self.num_boxes * 2]
        for box_id in range(self.num_boxes):
            box_pos = box_poses[box_id * 2 : (box_id + 1) * 2]
            if grid_map[box_pos[0],box_pos[1]] == 0:
                grid_map[box_pos[0],box_pos[1]] = 1
                continue
            else:
                return True

        ramp_poses = task[(self.num_hiders + self.num_seekers) * 2 + self.num_boxes * 3 : (self.num_hiders + self.num_seekers) * 2 + self.num_boxes * 3 + self.num_ramps * 2]
        for ramp_id in range(self.num_ramps):
            ramp_pos = ramp_poses[ramp_id * 2 : (ramp_id + 1) * 2]
            if grid_map[ramp_pos[0],ramp_pos[1]] == 0:
                grid_map[ramp_pos[0],ramp_pos[1]] = 1
                continue
            else:
                return True
        return False

    def mean_box_locked(self, batch_data):
        mean_box_lock = 0
        for data in batch_data:
            mean_box_lock += np.mean(data[self.num_hiders * 2 + self.num_seekers * 2 + self.num_boxes * 2 : self.num_hiders * 2 + self.num_seekers * 2 + self.num_boxes * 3])
        infos = {}
        infos['mean_box_lock'] = mean_box_lock / len(batch_data)
        return infos
        

