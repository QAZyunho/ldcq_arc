import os
import sys

curr_folder=os.path.abspath(__file__)
parent_folder=os.path.dirname(os.path.dirname(curr_folder))
sys.path.append(parent_folder) 

from argparse import ArgumentParser
import os
import pickle
# import gym
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.diffusion_models import (
    Model_mlp,
    Model_cnn_mlp,
    Model_Cond_Diffusion,
)
from models.skill_model import SkillModel
from utils.utils import get_dataset, ARC_Segment_Dataset

def collect_data(args):
    # dataset_file = parent_folder+'/data/'+args.env+'.pkl'
    # with open(dataset_file, "rb") as f:
    #     dataset = pickle.load(f)

    state_dim = args.h_dim
    a_dim = args.a_dim

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
        diffusion_steps=args.skill_model_diffusion_steps, 
        normalize_latent=args.normalize_latent,
        color_num=11,	# 0~9 색깔, 10은 배경
        action_num=args.a_dim,	# 36개의 action
        max_grid_size=args.max_grid_size,
    ).to(args.device)
    skill_model.load_state_dict(checkpoint['model_state_dict'])
    skill_model.eval()

    if args.do_diffusion:
        diffusion_nn_model = torch.load(os.path.join(args.checkpoint_dir, args.diffusion_model_filename)).to(args.device)
        
        diffusion_model = Model_Cond_Diffusion(
            diffusion_nn_model,
            betas=(1e-4, 0.02),
            n_T=args.diffusion_steps,
            device=args.device,
            x_dim=state_dim,
            y_dim=args.z_dim,
            drop_prob=None,
            guide_w=args.cfg_weight,
        )
        diffusion_model.eval()


    dataset = ARC_Segment_Dataset(
		data_path=args.solar_dir
	)
    len_train_dataset = dataset.__len__()
    print("Length dataset: {0}".format(len_train_dataset))
    
    train_loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=0)

    states_gt = np.zeros((len_train_dataset, 1, args.max_grid_size, args.max_grid_size))
    clip_gt = np.zeros((len_train_dataset, 1, args.max_grid_size, args.max_grid_size))
    in_grid_gt = np.zeros((len_train_dataset, 1, args.max_grid_size, args.max_grid_size))
    
    pair_in_gt = np.zeros((len_train_dataset, 3, args.max_grid_size, args.max_grid_size))
    pair_out_gt = np.zeros((len_train_dataset, 3, args.max_grid_size, args.max_grid_size))
    
    latent_gt = np.zeros((len_train_dataset, args.z_dim))
    if args.save_z_dist:
        latent_std_gt = np.zeros((len_train_dataset, args.z_dim))
    sT_gt = np.zeros((len_train_dataset, 1, args.max_grid_size, args.max_grid_size))
    clip_T_gt = np.zeros((len_train_dataset, 1, args.max_grid_size, args.max_grid_size))
    rewards_gt = np.zeros((len_train_dataset, 1))

    if args.do_diffusion:
        diffusion_latents_gt = np.zeros((len_train_dataset, args.num_diffusion_samples, args.z_dim))
        prior_latents_gt = np.zeros((len_train_dataset, args.num_prior_samples, args.z_dim))

    if not 'maze' in args.env and not 'kitchen' in args.env:
        terminals_gt = np.zeros((len_train_dataset, 1))
    gamma_array = np.power(args.gamma, np.arange(args.horizon))

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
        
        states_gt[start_idx : end_idx, 0] = state[:, 0, :, :].cpu().numpy()
        clip_gt[start_idx : end_idx, 0] = clip[:, 0, :, :].cpu().numpy()
        in_grid_gt[start_idx : end_idx, 0] = in_grid[:, 0, :, :].cpu().numpy()
        sT_gt[start_idx: end_idx, 0] = s_T[:, 0, :, :].cpu().numpy()
        clip_T_gt[start_idx: end_idx, 0] = clip_T[:, 0, :, :].cpu().numpy()
        
        pair_in_gt[start_idx: end_idx] = pair_in[:, :, :, :].cpu().numpy()
        pair_out_gt[start_idx: end_idx] = pair_out[:, :, :, :].cpu().numpy()
        
        rewards_gt[start_idx: end_idx, 0] = np.sum(reward.cpu().numpy()[:,:,0] * gamma_array, axis=1)
        terminals_gt[start_idx: end_idx] = np.sum(terminated.cpu().numpy(), axis=1)
        
        if not args.do_diffusion:
            with torch.no_grad():
                prior_latent_mean, prior_latent_std = skill_model.prior(state[:, 0:1, :], clip[:, 0:1, :], in_grid, pair_in, pair_out)   # 이게 맞나????
                prior_latent_mean = prior_latent_mean.repeat_interleave(args.num_prior_samples, 0)
                prior_latent_std = prior_latent_std.repeat_interleave(args.num_prior_samples, 0)
                
                prior_latents_gt[start_idx : end_idx] = torch.stack(torch.distributions.normal.Normal(prior_latent_mean, prior_latent_std).sample().chunk(state.shape[0])).cpu().numpy()
        else:
            diffusion_state = state[:, 0:1, :].repeat_interleave(args.num_diffusion_samples, 0)
            diffusion_clip = clip[:, 0:1, :].repeat_interleave(args.num_diffusion_samples, 0)
            diffusion_in_grid = in_grid[:, 0:1, :].repeat_interleave(args.num_diffusion_samples, 0)
            with torch.no_grad():
                diffusion_latents_gt[start_idx : end_idx] = torch.stack(diffusion_model.sample_extra(diffusion_state, diffusion_clip, diffusion_in_grid, predict_noise=args.predict_noise, extra_steps=args.extra_steps).chunk(state.shape[0])).cpu().numpy()

        output, output_std = skill_model.encoder(state, clip, in_grid, operation, selection, pair_in, pair_out)
        latent_gt[start_idx : end_idx] = output.detach().cpu().numpy().squeeze(1)
        if args.save_z_dist:
            latent_std_gt[start_idx : end_idx] = output_std.detach().cpu().numpy().squeeze(1)

    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)
    
    np.save(os.path.join(args.data_dir, args.skill_model_filename[:-4]+ '_states.npy'), states_gt)
    np.save(os.path.join(args.data_dir, args.skill_model_filename[:-4] + '_clip.npy'), clip_gt)
    np.save(os.path.join(args.data_dir, args.skill_model_filename[:-4] + '_in_grid.npy'), in_grid_gt)
    np.save(os.path.join(args.data_dir, args.skill_model_filename[:-4] + '_latents.npy'), latent_gt)
    np.save(os.path.join(args.data_dir, args.skill_model_filename[:-4] + '_sT.npy'), sT_gt)
    np.save(os.path.join(args.data_dir, args.skill_model_filename[:-4] + '_clip_T.npy'), clip_T_gt)
    np.save(os.path.join(args.data_dir, args.skill_model_filename[:-4] + '_rewards.npy'), rewards_gt)
    np.save(os.path.join(args.data_dir, args.skill_model_filename[:-4] + '_pair_in.npy'), pair_in_gt)
    np.save(os.path.join(args.data_dir, args.skill_model_filename[:-4] + '_pair_out.npy'), pair_out_gt)
    
    if args.do_diffusion:
        np.save(os.path.join(args.data_dir, args.skill_model_filename[:-4] + '_sample_latents.npy'), diffusion_latents_gt)
    else:
        np.save(os.path.join(args.data_dir, args.skill_model_filename[:-4] + '_prior_latents.npy'), prior_latents_gt)
    if args.save_z_dist:
        np.save(os.path.join(args.data_dir, args.skill_model_filename[:-4] + '_latents_std.npy'), latent_std_gt)
    if not 'maze' in args.env and not 'kitchen' in args.env:
        np.save(os.path.join(args.data_dir, args.skill_model_filename[:-4] + '_terminals.npy'), terminals_gt)


if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument('--env', type=str, default='antmaze-large-diverse-v2')
    parser.add_argument('--device', type=str, default='cuda')
    
    parser.add_argument('--solar_dir', type=str, default=None)
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--checkpoint_dir', type=str, default=parent_folder+'/checkpoints/')
    parser.add_argument('--skill_model_filename', type=str)
    parser.add_argument('--diffusion_model_filename', type=str)
    
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--append_goals', type=int, default=0)
    parser.add_argument('--save_z_dist', type=int, default=0)
    parser.add_argument('--cum_rewards', type=int, default=0)

    parser.add_argument('--do_diffusion', type=int, default=1)
    parser.add_argument('--num_diffusion_samples', type=int, default=300)
    parser.add_argument('--num_prior_samples', type=int, default=300)
    parser.add_argument('--diffusion_steps', type=int, default=500)
    parser.add_argument('--cfg_weight', type=float, default=0.0)
    parser.add_argument('--extra_steps', type=int, default=5)
    parser.add_argument('--predict_noise', type=int, default=0)

    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--horizon', type=int, default=5)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--a_dist', type=str, default='normal')
    parser.add_argument('--encoder_type', type=str, default='gru')
    parser.add_argument('--state_decoder_type', type=str, default='mlp')
    parser.add_argument('--policy_decoder_type', type=str, default='mlp')       # 원래는 autoregressive
    parser.add_argument('--per_element_sigma', type=int, default=1)
    
    parser.add_argument('--train_diffusion_prior', type=int, default=1)
    parser.add_argument('--conditional_prior', type=int, default=1)
    parser.add_argument('--normalize_latent', type=int, default=0)
    
    parser.add_argument('--skill_model_diffusion_steps', type=int, default=100)

    parser.add_argument('--a_dim', type=int, default=36)
    parser.add_argument('--h_dim', type=int, default=256)
    parser.add_argument('--z_dim', type=int, default=16)
    parser.add_argument('--max_grid_size', type=int, default=30)
    

    args = parser.parse_args()

    collect_data(args)
