import os
import sys
import datetime

curr_folder=os.path.abspath(__file__)
parent_folder=os.path.dirname(os.path.dirname(curr_folder))
sys.path.append(parent_folder) 

from argparse import ArgumentParser
# from comet_ml import Experiment

# import gym
import pickle
import numpy as np
from tqdm import tqdm
import torch
from per_utils import NaivePrioritizedBuffer, FixedPrioritizedBuffer
from torch.utils.data import Dataset, DataLoader

from models.diffusion_models import (
    Model_mlp,
    Model_cnn_mlp,
    Model_Cond_Diffusion,
)
from models.dqn import DDQN


class QLearningDataset(Dataset):
    def __init__(
        self, data_dir, filename, train_or_test="train", test_prop=0.1, sample_z=False
    ):
        # just load it all into RAM
        self.state_all = np.load(os.path.join(data_dir, filename + "_states.npy"), allow_pickle=True)
        self.clip_all = np.load(os.path.join(data_dir, filename + "_clip.npy"), allow_pickle=True)
        self.in_grid_all = np.load(os.path.join(data_dir, filename + "_in_grid.npy"), allow_pickle=True)
        self.latent_all = np.load(os.path.join(data_dir, filename + "_latents.npy"), allow_pickle=True)
        self.sT_all = np.load(os.path.join(data_dir, filename + "_sT.npy"), allow_pickle=True)
        self.clip_T_all = np.load(os.path.join(data_dir, filename + "_clip_T.npy"), allow_pickle=True)
        self.pair_in_all = np.load(os.path.join(data_dir, filename + "_pair_in.npy"), allow_pickle=True)
        self.pair_out_all = np.load(os.path.join(data_dir, filename + "_pair_out.npy"), allow_pickle=True)
        self.rewards_all = np.load(os.path.join(data_dir, filename + "_rewards.npy"), allow_pickle=True)#(4*np.load(os.path.join(data_dir, filename + "_rewards.npy"), allow_pickle=True) - 30*4*0.5)/10 #zero-centering
        self.sample_z = sample_z
        if sample_z:
            self.latent_all_std = np.load(os.path.join(data_dir, filename + "_latents_std.npy"), allow_pickle=True)
        
        n_train = int(self.state_all.shape[0] * (1 - test_prop))
        if train_or_test == "train":
            self.state_all = self.state_all[:n_train]
            self.clip_all = self.clip_all[:n_train]
            self.in_grid_all = self.in_grid_all[:n_train]
            self.latent_all = self.latent_all[:n_train]
            self.sT_all = self.sT_all[:n_train]
            self.clip_T_all = self.clip_T_all[:n_train]
            self.pair_in_all = self.pair_in_all[:n_train]
            self.pair_out_all = self.pair_out_all[:n_train]
            self.rewards_all = self.rewards_all[:n_train]
        elif train_or_test == "test":
            self.state_all = self.state_all[n_train:]
            self.clip_all = self.clip_all[n_train:]
            self.in_gird_all = self.in_gird_all[n_train:]
            self.latent_all = self.latent_all[n_train:]
            self.sT_all = self.sT_all[n_train:]
            self.clip_T_all = self.clip_T_all[n_train:]
            self.pair_in_all = self.pair_in_all[n_train:]
            self.pair_out_all = self.pair_out_all[n_train:]
            self.rewards_all = self.rewards_all[n_train:]
        else:
            raise NotImplementedError

    def __len__(self):
        return self.state_all.shape[0]

    def __getitem__(self, index):
        state = self.state_all[index]
        clip = self.clip_all[index]
        in_grid = self.in_grid_all[index]
        latent = self.latent_all[index]
        if self.sample_z:
            latent_std = self.latent_all_std[index]
            latent = np.random.normal(latent,latent_std)
        sT = self.sT_all[index]
        clip_T = self.clip_T_all[index]
        reward = self.rewards_all[index]
        pair_in = self.pair_in_all[index]
        pair_out = self.pair_out_all[index]

        # return (state, clip, latent, reward, sT, clip_T, reward, pair_in, pair_out)
        return (state, clip, in_grid, latent, sT, clip_T, reward, pair_in, pair_out)
        # return (state, latent, sT, reward)

def PER_buffer_filler(data_dir, filename, test_prop=0.1, sample_z=False, sample_max_latents=False, alpha=0.6, do_diffusion=1):
    # just load it all into RAM
    state_all = np.load(os.path.join(data_dir, filename + "_states.npy"), allow_pickle=True)
    clip_all = np.load(os.path.join(data_dir, filename + "_clip.npy"), allow_pickle=True)
    in_grid_all = np.load(os.path.join(data_dir, filename + "_in_grid.npy"), allow_pickle=True)
    latent_all = np.load(os.path.join(data_dir, filename + "_latents.npy"), allow_pickle=True)
    sT_all = np.load(os.path.join(data_dir, filename + "_sT.npy"), allow_pickle=True)
    clip_T_all = np.load(os.path.join(data_dir, filename + "_clip_T.npy"), allow_pickle=True)
    pair_in_all = np.load(os.path.join(data_dir, filename + "_pair_in.npy"), allow_pickle=True)
    pair_out_all = np.load(os.path.join(data_dir, filename + "_pair_out.npy"), allow_pickle=True)
    rewards_all = np.load(os.path.join(data_dir, filename + "_rewards.npy"), allow_pickle=True)#(4*np.load(os.path.join(data_dir, filename + "_rewards.npy"), allow_pickle=True) - 30*4*0.5)/10 #zero-centering
    
    if sample_z:
        latent_all_std = np.load(os.path.join(data_dir, filename + "_latents_std.npy"), allow_pickle=True)
        
    if sample_max_latents:
        if do_diffusion:
            max_latents = np.load(os.path.join(data_dir, filename + "_sample_latents.npy"), allow_pickle=True)
        else:
            max_latents = np.load(os.path.join(data_dir, filename + "_prior_latents.npy"), allow_pickle=True)
            
    if not 'maze' in filename and not 'kitchen' in filename:
        terminals_all = np.load(os.path.join(data_dir, filename + "_terminals.npy"), allow_pickle=True)
        # rewards_all = rewards_all/10
    
    n_train = int(state_all.shape[0] * (1 - test_prop))
    
    # PER is only for training
    state_all = state_all[:n_train]
    clip_all = clip_all[:n_train]
    in_grid_all = in_grid_all[:n_train]
    latent_all = latent_all[:n_train]
    sT_all = sT_all[:n_train]
    clip_T_all = clip_T_all[:n_train]
    rewards_all = rewards_all[:n_train]
    pair_in_all = pair_in_all[:n_train]
    pair_out_all = pair_out_all[:n_train]
    
    if not 'maze' in filename and not 'kitchen' in filename:
        terminals_all = terminals_all[:n_train]
    if sample_max_latents:
        max_latents_all = max_latents[:n_train]
    else:
        max_latents_all = None
    
    # load into PER buffer
    # replay_buffer = NaivePrioritizedBuffer(n_train, prob_alpha=alpha)
    replay_buffer = FixedPrioritizedBuffer(n_train, num_samples=args.total_prior_samples, z_dim=args.z_dim, max_grid_size=args.max_grid_size, prob_alpha=alpha)
    
    for i in tqdm(range(n_train)):
        replay_buffer.push(
            state_all[i], 
            clip_all[i],
            in_grid_all[i], 
            latent_all[i], 
            rewards_all[i], 
            sT_all[i], 
            clip_T_all[i], 
            terminals_all[i], 
            pair_in_all[i], 
            pair_out_all[i], 
            max_latents_all[i]
        )
        
    return replay_buffer, state_all.shape[-1], latent_all.shape[-1]

def train(args):
    # get datasets set up
    if args.per_buffer:
        # per_buffer, x_shape, y_dim = PER_buffer_filler(args.data_dir, args.skill_model_filename[:-4], test_prop=args.test_split, sample_z=args.sample_z, sample_max_latents=args.sample_max_latents, alpha=args.alpha, do_diffusion=args.do_diffusion)
        per_buffer, _, y_dim = PER_buffer_filler(args.data_dir, args.skill_model_filename[:-4], test_prop=args.test_split, sample_z=args.sample_z, sample_max_latents=args.sample_max_latents, alpha=args.alpha, do_diffusion=args.do_diffusion)
        x_shape = args.h_dim
    else:
        torch_data_train = QLearningDataset(
            args.data_dir, args.skill_model_filename[:-4], train_or_test="train", test_prop=args.test_split, sample_z=args.sample_z
        )
        x_shape = args.h_dim
        y_dim = torch_data_train.latent_all.shape[1]
        
        dataload_train = DataLoader(
            torch_data_train, batch_size=args.batch_size, shuffle=True, num_workers=0
        )
    '''

    torch_data_test = QLearningDataset(
        args.data_dir, args.skill_model_filename[:-4], train_or_test="test", test_prop=args.test_split, sample_z=args.sample_z
    )
    dataload_test = DataLoader(
        torch_data_test, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    '''
    # create model
    model = None
    if args.do_diffusion:
        # diffusion_nn_model = torch.load(os.path.join(args.checkpoint_dir, args.skill_model_filename[:-4] + '_diffusion_prior_best.pt')).to(args.device)
        diffusion_nn_model = torch.load(os.path.join(args.checkpoint_dir, args.diffusion_model_filename)).to(args.device)
        model = Model_Cond_Diffusion(
            diffusion_nn_model,
            betas=(1e-4, 0.02),
            n_T=args.diffusion_steps,
            device=args.device,
            x_dim=x_shape,
            y_dim=y_dim,
            drop_prob=args.drop_prob,
            guide_w=args.cfg_weight,
        ).to(args.device)
        model.eval()

    dqn_agent = DDQN(state_dim = x_shape, z_dim=y_dim, h_dim=args.h_dim, diffusion_prior=model, total_prior_samples=args.total_prior_samples, num_prior_samples=args.num_prior_samples, gamma=args.gamma,max_grid_size=args.max_grid_size,horizon=args.horizon)
    # dqn_agent.learn(dataload_train=per_buffer if args.per_buffer else dataload_train, n_epochs=args.n_epoch,
    #     diffusion_model_name=args.skill_model_filename[:-4], cfg_weight=args.cfg_weight, per_buffer = args.per_buffer, batch_size = args.batch_size, gpu_name=args.gpu_name)
    task_name = args.solar_dir.split("/")[-1]
    
    dqn_agent.learn(dataload_train=per_buffer, n_epochs=args.n_epoch,
        diffusion_model_name=args.skill_model_filename[:-4], cfg_weight=args.cfg_weight, per_buffer = args.per_buffer, batch_size = args.batch_size, gpu_name=args.gpu_name,q_checkpoint_dir=args.q_checkpoint_dir,task_name=task_name)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('--env', type=str, default='antmaze-large-diverse-v2')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--n_epoch', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--net_type', type=str, default='unet')
    parser.add_argument('--n_hidden', type=int, default=512)
    parser.add_argument('--test_split', type=float, default=0.0)
    parser.add_argument('--sample_z', type=int, default=0)
    parser.add_argument('--per_buffer', type=int, default=1)
    parser.add_argument('--sample_max_latents', type=int, default=1)
    parser.add_argument('--total_prior_samples', type=int, default=1000)
    parser.add_argument('--num_prior_samples', type=int, default=1000)

    parser.add_argument('--gamma', type=float, default=0.995)
    parser.add_argument('--alpha', type=float, default=0.7)

    parser.add_argument('--checkpoint_dir', type=str, default=parent_folder+'/checkpoints/')
    parser.add_argument('--q_checkpoint_dir', type=str, default=parent_folder+'/q_checkpoints/')
    parser.add_argument('--solar_dir', type=str, default=None)
    parser.add_argument('--data_dir', type=str, default=parent_folder+'/data/')
    parser.add_argument('--skill_model_filename', type=str)
    parser.add_argument('--diffusion_model_filename', type=str)

    parser.add_argument('--do_diffusion', type=int, default=1)
    parser.add_argument('--drop_prob', type=float, default=0.0)
    parser.add_argument('--diffusion_steps', type=int, default=100)
    parser.add_argument('--cfg_weight', type=float, default=0.0)
    parser.add_argument('--predict_noise', type=int, default=0)
    
    parser.add_argument('--a_dim', type=int, default=36)
    parser.add_argument('--z_dim', type=int, default=256)
    parser.add_argument('--h_dim', type=int, default=256)
    parser.add_argument('--s_dim', type=int, default=256)
    parser.add_argument('--horizon',type=int, default=5)
    parser.add_argument('--gpu_name', type=str, required=True)
    parser.add_argument('--max_grid_size', type=int, default=30)
    args = parser.parse_args()
    
    train(args)