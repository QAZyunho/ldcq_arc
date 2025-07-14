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
    
    skill_model = SkillModel(
                            state_dim,
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
                            conditional_prior=args.conditional_prior,
                            train_diffusion_prior=args.train_diffusion_prior,
                            diffusion_steps=args.diffusion_steps, 
                            normalize_latent=args.normalize_latent,
                            action_num=a_dim,
                            max_grid_size=args.max_grid_size,
                            use_in_out=args.use_in_out,
                            ).to(args.device)
    
    skill_model.load_state_dict(checkpoint['model_state_dict'])
    skill_model.eval()

    dataset = ARC_Segment_Dataset(data_path=args.solar_dir)

    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=8)
    
    len_train_dataset = dataset.__len__()
    
    states_gt = np.zeros((len_train_dataset, 1, args.max_grid_size, args.max_grid_size))
    clip_gt = np.zeros((len_train_dataset, 1, args.max_grid_size, args.max_grid_size))
    in_grid_gt = np.zeros((len_train_dataset, 1, args.max_grid_size, args.max_grid_size))
    latent_gt = np.zeros((len_train_dataset, args.z_dim))
    pair_in_gt = np.zeros((len_train_dataset, args.num_ex_pair, args.max_grid_size, args.max_grid_size))
    pair_out_gt = np.zeros((len_train_dataset, args.num_ex_pair, args.max_grid_size, args.max_grid_size))
    
    if args.save_z_dist:
        latent_std_gt = np.zeros((len_train_dataset, args.z_dim))
       
    
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))

    for batch_id, (state, s_T, clip, clip_T, selection, operation, reward, terminated, _, in_grid, out_grid, ex_in, ex_out) in pbar:
        state = state.to(args.device)
        clip = clip.to(args.device)
        in_grid = in_grid.to(args.device)
        selection = selection.to(args.device)
        operation = operation.to(args.device)
        terminated = terminated.to(args.device)
        pair_in = ex_in.to(args.device)
        pair_out = ex_out.to(args.device)
        s_T = s_T.to(args.device)
        clip_T = clip_T.to(args.device)

        start_idx = batch_id * args.batch_size
        end_idx = start_idx + args.batch_size
        
        states_gt[start_idx : end_idx, 0, :, :] = state[:, 0, :, :].cpu().numpy()
        clip_gt[start_idx : end_idx, 0, :, :] = clip[:, 0, :, :].cpu().numpy()
        in_grid_gt[start_idx : end_idx, 0, :, :] = in_grid[:, 0, :, :].cpu().numpy()
        pair_in_gt[start_idx: end_idx] = pair_in[:, :, :, :].cpu().numpy()
        pair_out_gt[start_idx: end_idx] = pair_out[:, :, :, :].cpu().numpy()
        
        
        output, output_std = skill_model.encoder(state, clip, in_grid, operation, selection, pair_in, pair_out)
        latent_gt[start_idx : end_idx] = output.detach().cpu().numpy().squeeze(1)

    if not os.path.exists(args.data_dir):
        # 디렉토리가 없으면 새로 만듭니다
        os.makedirs(args.data_dir)
        print(f"디렉토리가 생성되었습니다: {args.data_dir}")

    np.save(os.path.join(args.data_dir,f'{args.skill_model_filename[:-4]}_states.npy'), states_gt)
    np.save(os.path.join(args.data_dir,f'{args.skill_model_filename[:-4]}_latents.npy'), latent_gt)
    np.save(os.path.join(args.data_dir,f'{args.skill_model_filename[:-4]}_clip.npy'), clip_gt)
    np.save(os.path.join(args.data_dir,f'{args.skill_model_filename[:-4]}_in_grid.npy'), in_grid_gt)
    np.save(os.path.join(args.data_dir, args.skill_model_filename[:-4] + '_pair_in.npy'), pair_in_gt)
    np.save(os.path.join(args.data_dir, args.skill_model_filename[:-4] + '_pair_out.npy'), pair_out_gt)
    
    if args.save_z_dist:
        np.save(os.path.join(args.data_dir,f'{args.skill_model_filename[:-4]}_latents_std.npy'), latent_std_gt)

if __name__ == '__main__':

    parser = ArgumentParser()
     # #####해놓은 것들이 argument 잘못넣으면 안 돌아가는 것들, 돌리기 전 꼭 확인할 것
    parser.add_argument('--env', type=str, default='antmaze-large-diverse-v2') #####
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--checkpoint_dir', type=str, default=parent_folder+'/checkpoints')
    parser.add_argument('--data_dir', type=str, default=parent_folder+'/data')
    parser.add_argument('--solar_dir', type=str, default=None)
    parser.add_argument('--skill_model_filename', type=str) #####
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--append_goals', type=int, default=0) #####
    parser.add_argument('--save_z_dist', type=int, default=0)
    parser.add_argument('--get_rewards', type=int, default=1)
    
    parser.add_argument('--horizon', type=int, default=30)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--beta', type=float, default=0.05)
    parser.add_argument('--a_dist', type=str, default='normal')
    
    parser.add_argument('--encoder_type', type=str, default='gru') 
    parser.add_argument('--state_decoder_type', type=str, default='mlp') #####
    parser.add_argument('--policy_decoder_type', type=str, default='autoregressive') #####
    parser.add_argument('--per_element_sigma', type=int, default=1)
    parser.add_argument('--conditional_prior', type=int, default=0)
    parser.add_argument('--train_diffusion_prior', type=int, default=0)
    parser.add_argument('--h_dim', type=int, default=256)
    parser.add_argument('--z_dim', type=int, default=16)
    parser.add_argument('--a_dim', type=int, default=256)
    parser.add_argument('--s_dim', type=int, default=256)
    
    parser.add_argument('--normalize_latent', type=int, default=0)
    parser.add_argument('--diffusion_steps', type=int, default=100)
    parser.add_argument('--max_grid_size', type=int, default=30)
    parser.add_argument('--num_ex_pair', type=int, default=3)
    parser.add_argument('--use_in_out', type=int, default=0)  # 0: False, 1: True
    
    args = parser.parse_args()

    collect_data(args)
