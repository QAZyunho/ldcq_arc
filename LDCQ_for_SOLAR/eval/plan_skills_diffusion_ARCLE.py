import os
import sys

curr_folder=os.path.abspath(__file__)
parent_folder=os.path.dirname(os.path.dirname(curr_folder))
sys.path.append(parent_folder) 
from argparse import ArgumentParser

import numpy as np
import torch
import random
import gymnasium as gym
# import d4rl
import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('TkAgg')
# from mujoco_py import GlfwContext
# GlfwContext(offscreen=True)
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from models.diffusion_models import (
    Model_mlp,
    Model_cnn_mlp,
    Model_Cond_Diffusion,
)
from models.skill_model import SkillModel

import multiprocessing as mp
from arcle.loaders import Loader
from typing import Dict, List, Tuple
from numpy.typing import NDArray
from pathlib import Path
import json
import time
from tqdm import tqdm

class ARC_Dataloader(Loader):
    def __init__(self, data_path, train=True) -> None:
        self.data_path = data_path
        super().__init__(train=train)
        
    
    def get_path(self, **kwargs) -> List[str]:
        data_path = Path(self.data_path)

        pathlist = []

        for path, _, files in os.walk(data_path):
            for name in files:
                if 'expert' in name or 'gold_standard' in name:
                    pathlist.append(os.path.join(path, name))   

        self.num_dataset = len(pathlist)
        
        if(self.num_dataset == 0):
            raise ValueError("Wrong data path or empty folder. Please check the data path.")
        else:
            print("Number of episodes: {0}".format(self.num_dataset))
        
        # pathlist.sort()
        return pathlist

    def parse(self, **kwargs) -> List[Tuple[List[NDArray], List[NDArray], List[NDArray], List[NDArray], Dict]]:
        dat = []

        for p in self._pathlist:
            with open(p) as fp:
                trajectory = json.load(fp)

                ti: List[NDArray] = []
                to: List[NDArray] = []
                ei: List[NDArray] = []
                eo: List[NDArray] = []
                
                ti_h, ti_w = trajectory['grid_dim'][0]
                to_h, to_w = trajectory['grid_dim'][-1]
                
                # Single Task 경우
                ti.append(np.array(trajectory['in_grid'], dtype=np.int8)[:ti_h, :ti_w])
                to.append(np.array(trajectory['out_grid'], dtype=np.int8)[:to_h, :to_w])


                for i in range(len(trajectory['ex_in'])):
                    ei_h, ei_w = trajectory['ex_in_grid_dim'][i]
                    eo_h, eo_w = trajectory['ex_out_grid_dim'][i]
                    
                    ei.append(np.array(trajectory['ex_in'][i], dtype=np.int8)[:ei_h, :ei_w])
                    eo.append(np.array(trajectory['ex_out'][i], dtype=np.int8)[:eo_h, :eo_w])

                desc = {'id': trajectory['desc']['id'],
                        'ex_in_grid_dim': trajectory['ex_in_grid_dim'], 
                        'ex_out_grid_dim' : trajectory['ex_out_grid_dim'],
                    }

                dat.append((ei,eo,ti,to,desc))  # ARCLE 순서
                
        return self.convert_grid_to_uint8(dat)
        # return dat

    def convert_grid_to_uint8(self, item):
            if isinstance(item, tuple):
                return tuple(self.convert_grid_to_uint8(elem) for elem in item)
            elif isinstance(item, list):
                return [self.convert_grid_to_uint8(elem) for elem in item]
            elif isinstance(item, np.ndarray):
                return np.array([self.convert_grid_to_uint8(elem) for elem in item])
            elif isinstance(item, np.integer):
                return np.uint8(item)
            elif isinstance(item, np.floating):
                return np.uint8(item)
            else:
                return item

def sel_bbox_to_mask(selection_bbox, max_grid_size):
    x, y, h, w = selection_bbox
    sel_mask = np.zeros(max_grid_size, dtype=np.int8)
    sel_mask[x:x+h+1, y:y+w+1] = 1
    return sel_mask

def q_policy(
        diffusion_model,
        skill_model,
        state_0,
        clip_0,
        in_grid, 
        pair_in, 
        pair_out,
        state_mean,
        state_std,
        latent_mean,
        latent_std,
        num_parallel_envs,
        num_diffusion_samples,
        extra_steps,
        planning_depth,
        predict_noise,
        append_goals,
        dqn_agent,
    ):

    # state_dim = state_0.shape[1]
    state = state_0.repeat_interleave(num_diffusion_samples, 0).unsqueeze(1)
    clip = clip_0.repeat_interleave(num_diffusion_samples, 0).unsqueeze(1)
    in_grid_inter = in_grid.repeat_interleave(num_diffusion_samples, 0).unsqueeze(1)
    pair_in = pair_in.repeat_interleave(num_diffusion_samples, 0)
    pair_out = pair_out.repeat_interleave(num_diffusion_samples, 0)
    
    latent, q_vals = dqn_agent.get_max_skills(state.float(), clip.float(), in_grid_inter.float(), pair_in.float(), pair_out.float(), is_eval=True)

    best_latent = torch.zeros((num_parallel_envs, latent.shape[1])).to(args.device)

    for env_idx in range(num_parallel_envs):
        start_idx = env_idx * num_diffusion_samples
        end_idx = start_idx + num_diffusion_samples
        
        # top10_values, top10_indices = torch.topk(q_vals[start_idx:end_idx], k=5)
        # top_z = latent[start_idx + top10_indices].clone()
        # print(np.round(top10_values.clone().cpu().numpy().astype(np.float64), 4).tolist())
        # for z in top_z:
        #     operation, x, y, h, w = skill_model.decoder.ll_policy.tensor_policy(
        #                             state_0[env_idx].unsqueeze(0), clip_0[env_idx].unsqueeze(0), in_grid[env_idx].unsqueeze(0), z, pair_in[env_idx].unsqueeze(0), pair_out[env_idx].unsqueeze(0))
            
        #     operation = torch.argmax(operation.clone()).cpu().numpy()  # argmax 후 CPU로 옮겨 NumPy로 변환하고 스칼라로 만듦
        #     x = torch.argmax(x.clone()).cpu().numpy()  # argmax 후 NumPy 배열로 변환
        #     y = torch.argmax(y.clone()).cpu().numpy()  # argmax 후 NumPy 배열로 변환
        #     h = torch.argmax(h.clone()).cpu().numpy()  # argmax 후 NumPy 배열로 변환
        #     w = torch.argmax(w.clone()).cpu().numpy() 
                            
        #     print("op: {0}, x: {1}, y: {2}, h : {3}, w : {4}".format(operation, x, y, h, w))
        
        max_idx = torch.argmax(q_vals[start_idx:end_idx])
        best_latent[env_idx] = latent[start_idx + max_idx]

    return best_latent

def diffusion_prior_policy(
        diffusion_model,
        skill_model,
        state_0,
        clip_0,
        in_grid,
        pair_in, 
        pair_out,
        state_mean,
        state_std,
        latent_mean,
        latent_std,
        num_parallel_envs,
        num_diffusion_samples,
        extra_steps,
        planning_depth,
        predict_noise,
        append_goals,
        dqn_agent,
    ):

    state_dim = state_0.shape[1]
    state_0 = state_0.unsqueeze(1)

    clip_dim = clip_0.shape[1]
    clip_0 = clip_0.unsqueeze(1)
    
    in_grid_dim = in_grid.shape[1]
    in_grid_unsq = in_grid.unsqueeze(1)
    
    latent = diffusion_model.sample_extra((state_0 - state_mean) / state_std, (clip_0 - state_mean) / state_std, (in_grid_unsq - state_mean) / state_std, predict_noise=predict_noise, extra_steps=extra_steps) * latent_std + latent_mean
    
    return latent


def prior_policy(
        diffusion_model,
        skill_model,
        state_0,
        clip_0,
        in_grid,
        pair_in, 
        pair_out,
        state_mean,
        state_std,
        latent_mean,
        latent_std,
        num_parallel_envs,
        num_diffusion_samples,
        extra_steps,
        planning_depth,
        predict_noise,
        append_goals,
        dqn_agent,
    ):

    state_dim = state_0.shape[1]
    state_0 = state_0.unsqueeze(1)
    
    clip_dim = clip_0.shape[1]
    clip_0 = clip_0.unsqueeze(1)
    
    in_grid_dim = in_grid.shape[1]
    in_grid_unsq = in_grid.unsqueeze(1)
    
    latent, latent_prior_std = skill_model.prior(state_0, clip_0, in_grid_unsq, pair_in, pair_out)
    eps = torch.normal(torch.zeros(latent.size()).cuda(), torch.ones(latent.size()).cuda())
    
    return latent + latent_prior_std * eps
        
def eval_func(diffusion_model,
              skill_model,
              policy,
              envs,
              state_dim,
              state_mean,
              state_std,
              latent_mean,
              latent_std,
              num_evals,
              num_parallel_envs,
              num_diffusion_samples,
              extra_steps,
              planning_depth,
              exec_horizon,
              predict_noise,
              render,
              append_goals,
              dqn_agent=None,
              env_name=None,
              loader=None):
    
    print("Render mode : None")
    print(f"test_data: {args.test_solar_dir}")
    print(f"exec_horizon: {args.exec_horizon}")
    print(f"q_checkpoint_dir: {args.q_checkpoint_dir}")
    print(f"q_checkpoint_steps: {args.q_checkpoint_steps}")
    print(f"checkpoint_dir :{args.checkpoint_dir}")
    print(f"skill_model_filename :{args.skill_model_filename}")

    
    with torch.no_grad():
        assert num_evals % num_parallel_envs == 0
        num_evals = num_evals // num_parallel_envs

        score_submit = 0
        score_reach = 0
        
        # pbar = tqdm(range(num_evals))
        
        for eval_step in range(num_evals):
        # for eval_step in pbar:
            state_0 = torch.full((num_parallel_envs, args.max_grid_size, args.max_grid_size), 10).to(args.device)
            clip_0 = torch.full((num_parallel_envs, args.max_grid_size, args.max_grid_size), 10).to(args.device)
            in_grid = torch.full((num_parallel_envs, args.max_grid_size, args.max_grid_size), 10).to(args.device)
            
            pair_in = torch.full((num_parallel_envs, 3, args.max_grid_size, args.max_grid_size), 10).to(args.device)
            pair_out = torch.full((num_parallel_envs, 3, args.max_grid_size, args.max_grid_size), 10).to(args.device)
            
            done = [False] * num_parallel_envs
            reach_ans = [False] * num_parallel_envs
            count_none = [0] * num_parallel_envs
            for env_idx in range(len(envs)):
                # input-output pair 추출
                ex_in, ex_out, tt_in, tt_out, desc = loader.pick(data_index=eval_step)
                
                for i in range(3):
                    ei_h, ei_w = desc['ex_in_grid_dim'][i]
                    eo_h, eo_w = desc['ex_out_grid_dim'][i]
                
                    pair_in[env_idx, i, :ei_h, :ei_w] = torch.from_numpy(np.array(ex_in[i])).to(args.device)
                    pair_out[env_idx, i, :eo_h, :eo_w] = torch.from_numpy(np.array(ex_out[i])).to(args.device)


            # test input 추출
            obs, info = envs[env_idx].reset(options={'prob_index': eval_step, 'subprob_index': 0, 'adaptation':False})

            obs_x, obs_y = obs['grid_dim']
            clip_x, clip_y = obs['clip_dim']
            
            state_0[env_idx, :obs_x, :obs_y] = torch.from_numpy(obs['grid'][:obs_x, :obs_y].copy()).to(args.device)
            clip_0[env_idx, :clip_x, :clip_y] = torch.from_numpy(obs['clip'][:clip_x, :clip_y].copy()).to(args.device)
            in_grid[env_idx, :obs_x, :obs_y] = torch.from_numpy(obs['grid'][:obs_x, :obs_y].copy()).to(args.device)
            
            # state_0[env_idx] = torch.from_numpy(envs[env_idx].reset())

            env_step = 0
                
            if 'ARCLE/O2ARCv2Env-v0' in env_name:
                total_steps = 20
            else:
                ValueError("Only ARCLE!")

            print(f"id: {desc['id']}")
                
            while env_step < total_steps:
                # s0, clip0, pair_in, pair_out,
                best_latent = policy(
                                diffusion_model,
                                skill_model,
                                state_0,
                                clip_0,
                                in_grid,
                                pair_in, 
                                pair_out,
                                state_mean,
                                state_std,
                                latent_mean,
                                latent_std,
                                num_parallel_envs,
                                num_diffusion_samples,
                                extra_steps,
                                planning_depth,
                                predict_noise,
                                append_goals,
                                dqn_agent,
                )
                
                # best_latent = torch.randn(1, 1, 1, args.z_dim).to(args.device)

                for _ in range(exec_horizon):
                    for env_idx in range(len(envs)):
                        
                        # for sample_num in range(args.num_diffusion_samples):
                        if not done[env_idx]:
                            # print(state_0[env_idx].shape)
                            # print(clip_0[env_idx].shape)
                            # print(in_grid[env_idx].shape)
                            # print(best_latent[env_idx].shape)
                            # print(pair_in[env_idx].shape)
                            # print(pair_out[env_idx].shape)
                            operation, x, y, h, w = skill_model.decoder.ll_policy.tensor_policy(
                                state_0[env_idx].unsqueeze(0), clip_0[env_idx].unsqueeze(0), in_grid[env_idx].unsqueeze(0), best_latent[env_idx].unsqueeze(0), pair_in[env_idx].unsqueeze(0), pair_out[env_idx].unsqueeze(0))
                            
                            operation = torch.argmax(operation.clone()).cpu().numpy()  # argmax 후 CPU로 옮겨 NumPy로 변환하고 스칼라로 만듦
                            x = torch.argmax(x.clone()).cpu().numpy()  # argmax 후 NumPy 배열로 변환
                            y = torch.argmax(y.clone()).cpu().numpy()  # argmax 후 NumPy 배열로 변환
                            h = torch.argmax(h.clone()).cpu().numpy()  # argmax 후 NumPy 배열로 변환
                            w = torch.argmax(w.clone()).cpu().numpy() 
                            
                            # operation = np.argmax(operation)
                            # x = np.argmax(x)
                            # y = np.argmax(y)
                            # h = np.argmax(h)
                            # w = np.argmax(w)
                            
                            if(render != "ansi"):
                                if operation == 35:
                                    print("None")
                                    # count_none[env_idx] +=1
                                    # if count_none[env_idx] >= 10:
                                    #     break
                                    done[env_idx] = 1
                                    continue
                                    
                                    
                                print("Step: {0}| op: {1}, x: {2}, y: {3}, h : {4}, w : {5}".format(env_step, operation, x, y, h, w))
                            
                            # Operation에서 None 나오면 env에 안넣고 패스
                            if(operation == 35):
                                print("None !!")
                                done[env_idx] = 1
                                ValueError("잘하자")
                            else:
                                # time.sleep(1.0)
                                select = sel_bbox_to_mask((x, y, h, w), (args.max_grid_size, args.max_grid_size))
                                action = {'selection': select.astype(bool), 'operation': operation}
                                
                                try:
                                    obs, reward, done[env_idx], _, _ = envs[env_idx].step(action)
                                
                                    # time.sleep(2.0)
                                    if reward:
                                        score_submit += 1

                                    obs_x, obs_y  = obs['grid_dim']
                                    state_0[env_idx].fill_(10)
                                    state_0[env_idx, :obs_x, :obs_y] = torch.from_numpy(obs['grid'][:obs_x, :obs_y].copy())

                                    clip_x, clip_y  = obs['clip_dim']
                                    clip_0[env_idx].fill_(10)
                                    clip_0[env_idx, :clip_x, :clip_y] = torch.from_numpy(obs['clip'][:clip_x, :clip_y].copy())

                                    if np.array_equal(obs['grid'][:obs_x, :obs_y], tt_out[0]):
                                        reach_ans[env_idx] = True
                                        
                                except Exception as e:
                                    print("ARCLE execution error")
                                    continue
                                    
                            if render and env_idx == 0:
                                envs[env_idx].render()
                            
                            if(done[env_idx]):
                                print("Terminal!!")
                                print("================================================================")
                                # time.sleep(2.0)
                                break
                        
                    
                    env_step += 1
                    #print(env_step, score_submit)
                    if env_step > total_steps:
                        break
                if sum(done) == num_parallel_envs:
                    break
                
            for reach in reach_ans:
                if reach :
                    score_reach += 1 
                
            print("env_step : {0}".format(env_step))
            
            total_runs = (eval_step + 1) * num_parallel_envs
            
            print(f'Total score: {score_submit} out of {total_runs} i.e. Acc: {(score_submit / total_runs) * 100}%')
            print(f'Reach answer: {score_reach} out of {total_runs} i.e. Acc: {(score_reach / total_runs) * 100}%')

                # print(f'Total score: {score_submit} out of {total_runs} i.e. Acc: {(score_submit / total_runs) * 100}%')
            

def evaluate(args, loader):
    # env = gym.make(args.env, render_mode=None, data_loader=loader,
    #                max_grid_size=args.max_grid_size, colors=10, max_episode_steps=None, max_trial=3)
    
    # dataset = env.get_dataset()
    # state_dim = dataset['observations'].shape[1]
    # a_dim = dataset['actions'].shape[1]
    state_dim = args.s_dim
    a_dim = args.a_dim
    
    skill_model = SkillModel(state_dim,
                            a_dim,
                            args.z_dim,
                            args.h_dim,
                            horizon=args.horizon,
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
                            max_grid_size=args.max_grid_size
                            ).to(args.device)

    skill_model.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, args.skill_model_filename))['model_state_dict'])
    skill_model.eval()

    diffusion_model = None
    if not args.policy == 'prior':
        # if args.append_goals:
        #   diffusion_nn_model = torch.load(os.path.join(args.checkpoint_dir, args.skill_model_filename[:-4] + '_diffusion_prior_gc_best.pt')).to(args.device)
        # else:
        #   diffusion_nn_model = torch.load(os.path.join(args.checkpoint_dir, args.skill_model_filename[:-4] + '_diffusion_prior_best.pt')).to(args.device)
        diffusion_nn_model = torch.load(os.path.join(args.checkpoint_dir, args.diffusion_model_filename),weights_only=False).to(args.device)
        
        diffusion_model = Model_Cond_Diffusion(
            diffusion_nn_model,
            betas=(1e-4, 0.02),
            n_T=args.diffusion_steps,
            device=args.device,
            x_dim=state_dim + args.append_goals*2,
            y_dim=args.z_dim,
            drop_prob=None,
            guide_w=args.cfg_weight,
        )
        diffusion_model.eval()
            
        # envs = [gym.make(args.env, render_mode=None, data_loader=loader, max_grid_size=args.max_grid_size, colors=10, 
        #                  max_episode_steps=None, max_trial=3) for _ in range(args.num_parallel_envs)]

    if(args.render == 'ansi'):
        # envs = gym.make(args.env, data_loader=loader, render_mode='ansi', max_grid_size=(args.max_grid_size,args.max_grid_size), colors=10, max_trial=3)
        envs = [gym.make(args.env, data_loader=loader, render_mode='ansi', max_grid_size=(args.max_grid_size,args.max_grid_size), colors=10, max_trial=3)]
    else:
        envs = [gym.make(args.env, data_loader=loader, max_grid_size=(args.max_grid_size,args.max_grid_size), colors=10, max_trial=3)]
        
        
    if not args.append_goals:
        #state_all = np.load(os.path.join(args.test_solar_dir, args.skill_model_filename[:-4] + "_states.npy"), allow_pickle=True)
        state_mean = 0    #torch.from_numpy(state_all.mean(axis=0)).to(args.device).float()
        state_std = 1     #torch.from_numpy(state_all.std(axis=0)).to(args.device).float()

        #latent_all = np.load(os.path.join(args.test_solar_dir, args.skill_model_filename[:-4] + "_latents.npy"), allow_pickle=True)
        latent_mean = 0   #torch.from_numpy(latent_all.mean(axis=0)).to(args.device).float()
        latent_std = 1    #torch.from_numpy(latent_all.std(axis=0)).to(args.device).float()
    else:
        #state_all = np.load(os.path.join(args.test_solar_dir, args.skill_model_filename[:-4] + "_goals_states.npy"), allow_pickle=True)
        state_mean = 0    #torch.from_numpy(state_all.mean(axis=0)).to(args.device).float()
        state_std = 1     #torch.from_numpy(state_all.std(axis=0)).to(args.device).float()

        #latent_all = np.load(os.path.join(args.test_solar_dir, args.skill_model_filename[:-4] + "_goals_latents.npy"), allow_pickle=True)
        latent_mean = 0   #torch.from_numpy(latent_all.mean(axis=0)).to(args.device).float()
        latent_std = 1    #torch.from_numpy(latent_all.std(axis=0)).to(args.device).float()

    dqn_agent = None
    if args.policy == 'prior':
        policy_fn = prior_policy
    elif args.policy == 'diffusion_prior':
        policy_fn = diffusion_prior_policy
    elif args.policy == 'q':
        # dqn_agent = torch.load(os.path.join(args.q_checkpoint_dir, args.skill_model_filename[:-4]+'_dqn_agent_'+str(args.q_checkpoint_steps)+'_cfg_weight_'+str(args.cfg_weight)+'_PERbuffer.pt')).to(args.device)
        dqn_agent = torch.load(os.path.join(args.q_checkpoint_dir, args.skill_model_filename[:-4]+'_dqn_agent_'+str(args.q_checkpoint_steps)+'_cfg_weight_'+str(args.cfg_weight)+'_PERbuffer.pt'),weights_only=False).to(args.device)
        dqn_agent.diffusion_prior = diffusion_model
        dqn_agent.extra_steps = args.extra_steps
        dqn_agent.target_net_0 = dqn_agent.q_net_0
        dqn_agent.target_net_1 = dqn_agent.q_net_1
        dqn_agent.eval()
        dqn_agent.num_prior_samples = args.num_diffusion_samples
        policy_fn = q_policy
    else:
        raise NotImplementedError

    eval_func(diffusion_model,
                skill_model,
                policy_fn,
                envs,
                state_dim,
                state_mean,
                state_std,
                latent_mean,
                latent_std,
                args.num_evals,
                args.num_parallel_envs,
                args.num_diffusion_samples,
                args.extra_steps,
                args.planning_depth,
                args.exec_horizon,
                args.predict_noise,
                args.render,
                args.append_goals,
                dqn_agent,
                args.env,
                loader,
                )


if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument('--env', type=str, default='ARCLE/O2ARCv2Env-v0')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_evals', type=int, default=100)
    parser.add_argument('--num_parallel_envs', type=int, default=1)
    parser.add_argument('--checkpoint_dir', type=str, default=parent_folder+'/checkpoints')
    parser.add_argument('--q_checkpoint_dir', type=str, default=parent_folder+'/q_checkpoints')
    parser.add_argument('--q_checkpoint_steps', type=int, default=0)
    parser.add_argument('--test_solar_dir', type=str, default=parent_folder+'/data')
    parser.add_argument('--skill_model_filename', type=str)
    parser.add_argument('--diffusion_model_filename', type=str)
    parser.add_argument('--append_goals', type=int, default=0)

    parser.add_argument('--policy', type=str, default='q') #greedy/exhaustive/q
    parser.add_argument('--num_diffusion_samples', type=int, default=10)
    parser.add_argument('--diffusion_steps', type=int, default=100)
    parser.add_argument('--cfg_weight', type=float, default=0.0)
    parser.add_argument('--planning_depth', type=int, default=3)    # 이게 필요한가?
    parser.add_argument('--extra_steps', type=int, default=4)
    parser.add_argument('--predict_noise', type=int, default=0)
    parser.add_argument('--exec_horizon', type=int, default=1)

    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--a_dist', type=str, default='normal')
    parser.add_argument('--encoder_type', type=str, default='gru')
    parser.add_argument('--state_decoder_type', type=str, default='mlp')
    parser.add_argument('--policy_decoder_type', type=str, default='mlp')    # 원래는 'autoregressive'
    parser.add_argument('--per_element_sigma', type=int, default=1)
    parser.add_argument('--conditional_prior', type=int, default=1)
    parser.add_argument('--horizon', type=int, default=5)
    
    parser.add_argument('--normalize_latent', type=int, default=0)  # 원래는 0(바활성화)
    parser.add_argument('--train_diffusion_prior', type=int, default=0)
    parser.add_argument('--skill_model_diffusion_steps', type=int, default=500)
    
    parser.add_argument('--a_dim', type=int, default=36)
    parser.add_argument('--z_dim', type=int, default=16)
    parser.add_argument('--h_dim', type=int, default=256)
    parser.add_argument('--s_dim', type=int, default=256)
    
    parser.add_argument('--render', type=str, default=None)
    parser.add_argument('--max_grid_size', type=int, default=30)
    args = parser.parse_args()

    loader = ARC_Dataloader(data_path=args.test_solar_dir)
    evaluate(args, loader)
