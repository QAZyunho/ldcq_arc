import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader
import torch.distributions.normal as Normal
from torch.utils.data import Dataset
import json
from pathlib import Path
import ipdb
import random
import pickle
import os
import sys

curr_folder=os.path.abspath(__file__)
parent_folder=os.path.dirname(os.path.dirname(curr_folder))

def reparameterize(mean, std):
    eps = torch.normal(torch.zeros(mean.size()).cuda(), torch.ones(mean.size()).cuda())
    return mean + std*eps

def stable_weighted_log_sum_exp(x,w,sum_dim):
    a = torch.min(x)
    ipdb.set_trace()

    weighted_sum = torch.sum(w * torch.exp(x - a),sum_dim)

    return a + torch.log(weighted_sum)

def chunks(obs,actions,H,stride):
    '''
    obs is a N x 4 array
    goals is a N x 2 array
    H is length of chunck
    stride is how far we move between chunks.  So if stride=H, chunks are non-overlapping.  If stride < H, they overlap
    '''
    
    obs_chunks = []
    action_chunks = []
    N = obs.shape[0]
    for i in range(N//stride - H):
        start_ind = i*stride
        end_ind = start_ind + H
        
        obs_chunk = torch.tensor(obs[start_ind:end_ind,:],dtype=torch.float32)

        action_chunk = torch.tensor(actions[start_ind:end_ind,:],dtype=torch.float32)
        
        loc_deltas = obs_chunk[1:,:2] - obs_chunk[:-1,:2] #Franka or Maze2d
        
        norms = np.linalg.norm(loc_deltas,axis=-1)
        #USE VALUE FOR THRESHOLD CONDITION BASED ON ENVIRONMENT
        if np.all(norms <= 0.8): #Antmaze large 0.8 medium 0.67 / Franka 0.23 mixed/complete 0.25 partial / Maze2d 0.22
            obs_chunks.append(obs_chunk)
            action_chunks.append(action_chunk)
        else:
            pass

    print('len(obs_chunks): ',len(obs_chunks))
    print('len(action_chunks): ',len(action_chunks))
            
    return torch.stack(obs_chunks),torch.stack(action_chunks)


def get_dataset(env_name, horizon, stride, test_split=0.2, append_goals=False, get_rewards=False, separate_test_trajectories=False, cum_rewards=True):
    dataset_file = parent_folder+'/data/'+env_name+'.pkl'
    
    with open(dataset_file, "rb") as f:
        dataset = pickle.load(f)

    observations = []
    actions = []
    terminals = []
    if get_rewards:
        rewards = []
    # goals = []

    if env_name == 'antmaze-large-diverse-v2' or env_name == 'antmaze-medium-diverse-v2':

        num_trajectories = np.where(dataset['timeouts'])[0].shape[0]
        assert num_trajectories == 999, 'Dataset has changed. Review the dataset extraction'

        if append_goals:
            dataset['observations'] = np.hstack([dataset['observations'],dataset['infos/goal']])
        print('Total trajectories: ', num_trajectories)

        for traj_idx in range(num_trajectories):
            start_idx = traj_idx * 1001
            end_idx = (traj_idx + 1) * 1001

            obs = dataset['observations'][start_idx : end_idx]
            act = dataset['actions'][start_idx : end_idx]
            if get_rewards:
                rew = np.expand_dims(dataset['rewards'][start_idx : end_idx],axis=1)
                
            # reward = dataset['rewards'][start_idx : end_idx]
            # goal = dataset['infos/goal'][start_idx : end_idx]

            num_observations = obs.shape[0]

            for chunk_idx in range(num_observations // stride - horizon):
                chunk_start_idx = chunk_idx * stride
                chunk_end_idx = chunk_start_idx + horizon

                observations.append(torch.tensor(obs[chunk_start_idx : chunk_end_idx], dtype=torch.float32))
                actions.append(torch.tensor(act[chunk_start_idx : chunk_end_idx], dtype=torch.float32))
                if get_rewards:
                    if np.sum(rew[chunk_start_idx : chunk_end_idx]>0):
                        rewards.append(torch.ones((chunk_end_idx-chunk_start_idx,1), dtype=torch.float32))
                        break
                    rewards.append(torch.tensor(rew[chunk_start_idx : chunk_end_idx], dtype=torch.float32))
                # goals.append(torch.tensor(goal[chunk_start_idx : chunk_end_idx], dtype=torch.float32))

        observations = torch.stack(observations)
        actions = torch.stack(actions)
        if get_rewards:
            rewards = torch.stack(rewards)
        # goals = torch.stack(goals)

        num_samples = observations.shape[0]
        # print(num_samples)
        # assert num_samples == 960039, 'Dataset has changed. Review the dataset extraction'

        print('Total data samples extracted: ', num_samples)
        num_test_samples = int(test_split * num_samples)

        if separate_test_trajectories:
            train_indices = np.arange(0, num_samples - num_test_samples)
            test_indices = np.arange(num_samples - num_test_samples, num_samples)
        else:
            test_indices = np.random.choice(np.arange(num_samples), num_test_samples, replace=False)
            train_indices = np.array(list(set(np.arange(num_samples)) - set(test_indices)))
        np.random.shuffle(train_indices)

        observations_train = observations[train_indices]
        actions_train = actions[train_indices]
        if get_rewards:
            rewards_train = rewards[train_indices]
        else:
            rewards_train = None
        # goals_train = goals[train_indices]

        observations_test = observations[test_indices]
        actions_test = actions[test_indices]
        if get_rewards:
            rewards_test = rewards[test_indices]
        else:
            rewards_test = None
        # goals_test = goals[test_indices]

        return dict(observations_train=observations_train,
                    actions_train=actions_train,
                    rewards_train=rewards_train,
                    # goals_train=goals_train,
                    observations_test=observations_test,
                    actions_test=actions_test,
                    rewards_test=rewards_test,
                    # goals_test=goals_test,
                    )

    elif 'kitchen' in env_name:

        num_trajectories = np.where(dataset['terminals'])[0].shape[0]

        print('Total trajectories: ', num_trajectories)

        terminals = np.where(dataset['terminals'])[0]
        terminals = np.append(-1, terminals)

        for traj_idx in range(1, len(terminals)):
            start_idx = terminals[traj_idx - 1] + 1
            end_idx = terminals[traj_idx] + 1

            obs = dataset['observations'][start_idx : end_idx]
            act = dataset['actions'][start_idx : end_idx]
            rew = np.expand_dims(dataset['rewards'][start_idx : end_idx],axis=1)

            # reward = dataset['rewards'][start_idx : end_idx]
            # goal = dataset['infos/goal'][start_idx : end_idx]

            num_observations = obs.shape[0]

            for chunk_idx in range(num_observations // stride - horizon):
                chunk_start_idx = chunk_idx * stride
                chunk_end_idx = chunk_start_idx + horizon

                observations.append(torch.tensor(obs[chunk_start_idx : chunk_end_idx], dtype=torch.float32))
                actions.append(torch.tensor(act[chunk_start_idx : chunk_end_idx], dtype=torch.float32))
                if cum_rewards:
                    rewards.append(torch.tensor(rew[chunk_start_idx : chunk_end_idx], dtype=torch.float32))
                else:
                    rewards.append(torch.tensor(np.diff(rew[chunk_start_idx : chunk_end_idx], axis=0, prepend=rew[chunk_start_idx, 0]), dtype=torch.float32))
                # goals.append(torch.tensor(goal[chunk_start_idx : chunk_end_idx], dtype=torch.float32))

        observations = torch.stack(observations)
        actions = torch.stack(actions)
        rewards = torch.stack(rewards)

        num_samples = observations.shape[0]

        print('Total data samples extracted: ', num_samples)
        num_test_samples = int(test_split * num_samples)

        if separate_test_trajectories:
            train_indices = np.arange(0, num_samples - num_test_samples)
            test_indices = np.arange(num_samples - num_test_samples, num_samples)
        else:
            test_indices = np.random.choice(np.arange(num_samples), num_test_samples, replace=False)
            train_indices = np.array(list(set(np.arange(num_samples)) - set(test_indices)))
        np.random.shuffle(train_indices)

        observations_train = observations[train_indices]
        actions_train = actions[train_indices]
        rewards_train = rewards[train_indices]

        observations_test = observations[test_indices]
        actions_test = actions[test_indices]
        rewards_test = rewards[test_indices]

        return dict(observations_train=observations_train,
                    actions_train=actions_train,
                    rewards_train=rewards_train,
                    observations_test=observations_test,
                    actions_test=actions_test,
                    rewards_test=rewards_test,
                    )

    elif 'maze2d' in env_name:

        if append_goals:
            dataset['observations'] = np.hstack([dataset['observations'], dataset['infos/goal']])

        obs = dataset['observations']
        act = dataset['actions']

        if get_rewards:
            rew = np.expand_dims(dataset['rewards'], axis=1)

        # reward = dataset['rewards'][start_idx : end_idx]
        # goal = dataset['infos/goal'][start_idx : end_idx]

        num_observations = obs.shape[0]

        for chunk_idx in range(num_observations // stride - horizon):
            chunk_start_idx = chunk_idx * stride
            chunk_end_idx = chunk_start_idx + horizon

            observations.append(torch.tensor(obs[chunk_start_idx : chunk_end_idx], dtype=torch.float32))
            actions.append(torch.tensor(act[chunk_start_idx : chunk_end_idx], dtype=torch.float32))
            if get_rewards:
                rewards.append(torch.tensor(rew[chunk_start_idx : chunk_end_idx], dtype=torch.float32))
            # goals.append(torch.tensor(goal[chunk_start_idx : chunk_end_idx], dtype=torch.float32))

        observations = torch.stack(observations)
        actions = torch.stack(actions)
        if get_rewards:
            rewards = torch.stack(rewards)
        # goals = torch.stack(goals)

        num_samples = observations.shape[0]

        print('Total data samples extracted: ', num_samples)
        num_test_samples = int(test_split * num_samples)

        if separate_test_trajectories:
            train_indices = np.arange(0, num_samples - num_test_samples)
            test_indices = np.arange(num_samples - num_test_samples, num_samples)
        else:
            test_indices = np.random.choice(np.arange(num_samples), num_test_samples, replace=False)
            train_indices = np.array(list(set(np.arange(num_samples)) - set(test_indices)))
        np.random.shuffle(train_indices)

        observations_train = observations[train_indices]
        actions_train = actions[train_indices]
        if get_rewards:
            rewards_train = rewards[train_indices]
        else:
            rewards_train = None
        # goals_train = goals[train_indices]

        observations_test = observations[test_indices]
        actions_test = actions[test_indices]
        if get_rewards:
            rewards_test = rewards[test_indices]
        else:
            rewards_test = None
        # goals_test = goals[test_indices]

        return dict(observations_train=observations_train,
                    actions_train=actions_train,
                    rewards_train=rewards_train,
                    # goals_train=goals_train,
                    observations_test=observations_test,
                    actions_test=actions_test,
                    rewards_test=rewards_test,
                    # goals_test=goals_test,
                    )

    else:
        obs = dataset['observations']
        act = dataset['actions']
        rew = np.expand_dims(dataset['rewards'],axis=1)
        dones = np.expand_dims(dataset['terminals'],axis=1)
        episode_step = 0
        chunk_idx = 0

        while chunk_idx < rew.shape[0]-horizon+1:
            chunk_start_idx = chunk_idx
            chunk_end_idx = chunk_start_idx + horizon

            observations.append(torch.tensor(obs[chunk_start_idx : chunk_end_idx], dtype=torch.float32))
            actions.append(torch.tensor(act[chunk_start_idx : chunk_end_idx], dtype=torch.float32))
            rewards.append(torch.tensor(rew[chunk_start_idx : chunk_end_idx], dtype=torch.float32))
            terminals.append(torch.tensor(dones[chunk_start_idx : chunk_end_idx], dtype=torch.float32))
            if np.sum(dones[chunk_start_idx : chunk_end_idx]>0):
                episode_step = 0
                chunk_idx += horizon
            elif(episode_step==1000-horizon):
                episode_step = 0
                chunk_idx += horizon
            else:
                episode_step += 1
                chunk_idx += 1

        observations = torch.stack(observations)
        actions = torch.stack(actions)
        rewards = torch.stack(rewards)
        terminals = torch.stack(terminals)

        num_samples = observations.shape[0]

        print('Total data samples extracted: ', num_samples)
        num_test_samples = int(test_split * num_samples)

        if separate_test_trajectories:
            train_indices = np.arange(0, num_samples - num_test_samples)
            test_indices = np.arange(num_samples - num_test_samples, num_samples)
        else:
            test_indices = np.random.choice(np.arange(num_samples), num_test_samples, replace=False)
            train_indices = np.array(list(set(np.arange(num_samples)) - set(test_indices)))
        np.random.shuffle(train_indices)

        observations_train = observations[train_indices]
        actions_train = actions[train_indices]
        rewards_train = rewards[train_indices]
        terminals_train = terminals[train_indices]

        observations_test = observations[test_indices]
        actions_test = actions[test_indices]
        rewards_test = rewards[test_indices]
        terminals_test = terminals[test_indices]

        return dict(observations_train=observations_train,
                    actions_train=actions_train,
                    rewards_train=rewards_train,
                    terminals_train=terminals_train,
                    observations_test=observations_test,
                    actions_test=actions_test,
                    rewards_test=rewards_test,
                    terminals_test=terminals_test
                    )


class ARC_Segment_Dataset(Dataset):
    
    def __init__(self, data_path): 
        self.data_path = Path(data_path)

        self.file_list = []

        for path, _, files in os.walk(self.data_path):
            for name in files:
                self.file_list.append(os.path.join(path, name))

        self.num_dataset = len(self.file_list)
        
        if(self.num_dataset == 0):
            raise ValueError("Wrong data path or empty folder. Please check the data path.")
        else:
            print("Number of episodes: {0}".format(self.num_dataset))
        
    def __len__(self):
        return self.num_dataset

    def __getitem__(self, idx):
        trace_file = self.file_list[idx]
            
        with open(trace_file, 'r') as f:
            trace = json.load(f)
        
        # state = torch.LongTensor(trace['grid']).flatten(start_dim=1, end_dim=2)
        state = torch.FloatTensor(trace['grid'])
        selection = torch.LongTensor(trace['selection']).unsqueeze(-1)
        operation = torch.LongTensor(trace['operation']).unsqueeze(-1)
        reward = torch.FloatTensor(trace['reward'])
        terminated = torch.LongTensor(trace['terminated'])
        selection_mask = torch.LongTensor(trace['selection_mask'])
        
        in_grid = torch.FloatTensor(trace['in_grid']).unsqueeze(0)
        out_grid = torch.FloatTensor(trace['out_grid']).unsqueeze(0)
        
        clip = torch.FloatTensor(trace['clip'])
        clip_dim = torch.LongTensor(trace['clip_dim'])
        
        ex_in = torch.FloatTensor(trace['ex_in'])
        ex_out = torch.FloatTensor(trace['ex_out'])
        
        s_T= torch.FloatTensor(trace['next_grid']).unsqueeze(0)
        clip_T = torch.FloatTensor(trace['next_clip']).unsqueeze(0)
        
        return state, s_T, clip, clip_T, selection, operation, reward, terminated, selection_mask, in_grid, out_grid, ex_in, ex_out
    
if __name__ == "__main__":
    from tqdm import tqdm
    
    dataset = ARC_Segment_Dataset(
		data_path="/data/beomi/jaehyun/ARC_single/segment/test.10.09.12"
	)
    
    train_loader = DataLoader(
	dataset=dataset,
	batch_size=16,
	num_workers=8,
	shuffle=True)
    
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))

	# state, selection, operation, reward, terminated, selection_mask, in_grid, out_grid
    for _, (state, s_T, clip, clip_T, selection, operation, reward, terminated, selection_mask, in_grid, out_grid, ex_in, ex_out) in pbar:
        print("state shape: {0}".format(state.shape))
        print("selection shape: {0}".format(selection.shape))
        print("operation shape: {0}".format(operation.shape))
        print("reward shape: {0}".format(reward.shape))
        print("terminated shape: {0}".format(terminated.shape))
        print("selection_mask shape: {0}".format(selection_mask.shape))
        print("in_grid shape: {0}".format(in_grid.shape))
        print("out_grid shape: {0}".format(out_grid.shape))
        print("s_T shape: {0}".format(in_grid.shape))
        print("cip_T shape: {0}".format(out_grid.shape))
        print("ex_in shape: {0}".format(ex_in.shape))
        print("ex_out shape: {0}".format(ex_out.shape))
        print("굳!")
        break