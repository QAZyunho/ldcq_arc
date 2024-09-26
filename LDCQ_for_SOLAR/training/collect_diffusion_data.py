import os
import sys

curr_folder=os.path.abspath(__file__)
parent_folder=os.path.dirname(os.path.dirname(curr_folder))
sys.path.append(parent_folder) 

from argparse import ArgumentParser

from tqdm import tqdm
# import gym
import numpy as np
import torch
from torch.utils.data import DataLoader

from models.skill_model import SkillModel
from utils.utils import get_dataset, ARC_Segment_Dataset

def collect_data(args):
    if 'ARCLE' in args.env:
        state_dim = args.s_dim
        a_dim = args.a_dim
    else:
        raise NotImplementedError
    
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)
        
    skill_model_path = os.path.join(args.checkpoint_dir, args.skill_model_filename)

    checkpoint = torch.load(skill_model_path)

    skill_model = SkillModel(state_dim,
                             a_dim,
                             args.z_dim,
                             args.h_dim,
                             args.horizon,
                             a_dist=args.a_dist,
                             beta=args.beta,
                             fixed_sig=None,
                             encoder_type=args.encoder_type,
                             state_decoder_type=args.state_decoder_type,
                             policy_decoder_type=args.policy_decoder_type,
                             per_element_sigma=args.per_element_sigma,
                             action_num=args.a_dim,
                             conditional_prior=args.conditional_prior,
                             train_diffusion_prior=args.train_diffusion_prior,
                             normalize_latent=args.normalize_latent,
                             diffusion_steps=args.diffusion_steps,
                             max_grid_size=args.max_grid_size,
                             ).to(args.device)
 
    skill_model.load_state_dict(checkpoint['model_state_dict'])
    skill_model.eval()

    # if 'halfcheetah' in args.env or 'walker2d' in args.env: 
    #     dataset = get_dataset(args.env, args.horizon, args.stride, 0.0, args.append_goals,args.get_rewards)
    # else:
    #     dataset = get_dataset(args.env, args.horizon, args.stride, 0.0, args.append_goals)

    # obs_chunks_train = dataset['observations_train']
    # action_chunks_train = dataset['actions_train']
    
    # inputs_train = torch.cat([obs_chunks_train, action_chunks_train], dim=-1)
    
    dataset = ARC_Segment_Dataset(
	    data_path=args.solar_dir
    )

    train_loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=0,
	)

    len_train_dataset = dataset.__len__()
    print("Length dataset: {0}".format(len_train_dataset))
    
    # inputs_train = torch.cat([obs_chunks_train, action_chunks_train], dim=-1)

    # states_gt = np.zeros((inputs_train.shape[0], state_dim))
    # latent_gt = np.zeros((inputs_train.shape[0], args.z_dim))
    
    states_gt = np.zeros((len_train_dataset, 1, args.max_grid_size, args.max_grid_size))    # 맨 앞에 state
    clip_gt = np.zeros((len_train_dataset, 1, args.max_grid_size, args.max_grid_size))      # 맨 앞에 clip
    in_grid_gt = np.zeros((len_train_dataset, 1, args.max_grid_size, args.max_grid_size))   # Episode 첫번째 grid
    
    sT_gt = np.zeros((len_train_dataset, 1, args.max_grid_size, args.max_grid_size))        # 맨 뒤에 state
    clip_T_gt = np.zeros((len_train_dataset, 1, args.max_grid_size, args.max_grid_size))    # 맨 뒤에 clip
    latent_gt = np.zeros((len_train_dataset, args.z_dim))
    
    if args.save_z_dist:
        # latent_std_gt = np.zeros((inputs_train.shape[0], args.z_dim))
        latent_std_gt = np.zeros((len_train_dataset, args.z_dim))
    # sT_gt = np.zeros((inputs_train.shape[0], state_dim))
  
    
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))

    for batch_id, (state, s_T, clip, clip_T, selection, operation, reward, terminated, _, in_grid, out_grid, ex_in, ex_out) in pbar:
    # for batch_id, data in enumerate(train_loader):
        # data = data.to(args.device)
        # states = data[:, :, :skill_model.state_dim]
        # actions = data[:, :, skill_model.state_dim:]
        
        state = state.to(args.device)
        clip = clip.to(args.device)
        in_grid = in_grid.to(args.device)
        selection = selection.to(args.device)
        operation = operation.to(args.device)
        pair_in = ex_in.to(args.device)
        pair_out = ex_out.to(args.device)

        start_idx = batch_id * args.batch_size
        end_idx = start_idx + args.batch_size
        
        states_gt[start_idx : end_idx, 0] = state[:, 0, :, :].cpu().numpy()     # (Batch, 처음, 30, 30)
        clip_gt[start_idx : end_idx, 0] = clip[:, 0, :, :].cpu().numpy()        # (Batch, 처음, 30, 30)
        in_grid_gt[start_idx : end_idx, 0] = in_grid[:, 0, :, :].cpu().numpy()  # (Batch, 1, 30, 30)
        
        sT_gt[start_idx : end_idx, 0] = s_T[:, 0, :, :].cpu().numpy()        # (Batch, 마지막, 30, 30)
        clip_T_gt[start_idx : end_idx, 0] = clip_T[:, 0, :, :].cpu().numpy()     # (Batch, 마지막, 30, 30)
        
        output, output_std = skill_model.encoder(state, clip, in_grid, operation, selection, pair_in, pair_out)   # skill_model.encoder(states, actions, selection)
        latent_gt[start_idx : end_idx] = output.detach().cpu().numpy().squeeze(1)
        
        if args.save_z_dist:
            latent_std_gt[start_idx : end_idx] = output_std.detach().cpu().numpy().squeeze(1)

    np.save(os.path.join(args.data_dir, args.skill_model_filename[:-4] + '_states.npy'), states_gt)
    np.save(os.path.join(args.data_dir, args.skill_model_filename[:-4] + '_clip.npy'), clip_gt)
    np.save(os.path.join(args.data_dir, args.skill_model_filename[:-4] + '_in_grid.npy'), in_grid_gt)
    np.save(os.path.join(args.data_dir, args.skill_model_filename[:-4] + '_latents.npy'), latent_gt)
    np.save(os.path.join(args.data_dir, args.skill_model_filename[:-4] + '_sT.npy'), sT_gt)
    np.save(os.path.join(args.data_dir, args.skill_model_filename[:-4] + '_clip_T.npy'), clip_T_gt)
    if args.save_z_dist:
        np.save(os.path.join(args.data_dir, args.skill_model_filename[:-4] + '_latents_std.npy'), latent_std_gt)

if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument('--env', type=str, default='antmaze-large-diverse-v2')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--checkpoint_dir', type=str, default=parent_folder+'/checkpoints')
    parser.add_argument('--skill_model_filename', type=str)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--append_goals', type=int, default=0)
    parser.add_argument('--save_z_dist', type=int, default=0)
    parser.add_argument('--get_rewards', type=int, default=1)
    
    parser.add_argument('--horizon', type=int, default=5)       # 원래는 30
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--beta', type=float, default=0.05)
    parser.add_argument('--a_dist', type=str, default='normal')
    parser.add_argument('--encoder_type', type=str, default='gru')
    parser.add_argument('--state_decoder_type', type=str, default='mlp')
    parser.add_argument('--policy_decoder_type', type=str, default='mlp')   # 원래는 'autoregressive'
    parser.add_argument('--per_element_sigma', type=int, default=1)
    
    parser.add_argument('--conditional_prior', type=int, default=1)
    parser.add_argument('--normalize_latent', type=int, default=0)
    parser.add_argument('--train_diffusion_prior', type=int, default=0)
    parser.add_argument('--diffusion_steps', type=int, default=500)
    
    parser.add_argument('--a_dim', type=int, default=36)
    parser.add_argument('--z_dim', type=int, default=128)            # 원래는 16
    parser.add_argument('--h_dim', type=int, default=256)
    parser.add_argument('--s_dim', type=int, default=256)
    
    parser.add_argument('--solar_dir', type=str, default=None)
    parser.add_argument('--data_dir', type=str, default=parent_folder+'/data')
    parser.add_argument('--max_grid_size', type=int, default=30)
    
    args = parser.parse_args()

    collect_data(args)
