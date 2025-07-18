import os
import sys
import datetime

curr_folder=os.path.abspath(__file__)
parent_folder=os.path.dirname(os.path.dirname(curr_folder))
sys.path.append(parent_folder) 

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import numpy as np
import random
from models.q_learning_models import MLP_Q
# from comet_ml import Experiment
import copy
import wandb
from tqdm import tqdm

class DDQN(nn.Module):
    def __init__(self, state_dim, z_dim, h_dim=256, gamma=0.995, tau=0.995, lr=1e-3, num_prior_samples=100, total_prior_samples=100, extra_steps=0, horizon=10,device='cuda', diffusion_prior=None,max_grid_size=30):
        super(DDQN,self).__init__()

        self.state_dim = state_dim
        self.z_dim = z_dim
        self.gamma = gamma
        self.lr = lr
        self.num_prior_samples = num_prior_samples
        self.total_prior_samples = total_prior_samples
        self.extra_steps = extra_steps
        self.device = device
        self.tau = tau
        self.diffusion_prior = diffusion_prior
        self.horizon = horizon
        self.max_grid_size = max_grid_size
        self.q_net_0 = MLP_Q(state_dim=state_dim,z_dim=z_dim,h_dim=h_dim,max_grid_size=max_grid_size).to(self.device)
        self.q_net_1 = MLP_Q(state_dim=state_dim,z_dim=z_dim,h_dim=h_dim,max_grid_size=max_grid_size).to(self.device)
        self.target_net_0 = None
        self.target_net_1 = None
        
        # self.optimizer_0 = optim.Adam(params=self.q_net_0.parameters(), lr=lr)
        # self.optimizer_1 = optim.Adam(params=self.q_net_1.parameters(), lr=lr)
        self.optimizer_0 = optim.AdamW(params=self.q_net_0.parameters(), lr=lr)
        self.optimizer_1 = optim.AdamW(params=self.q_net_1.parameters(), lr=lr)
        self.scheduler_0 = optim.lr_scheduler.StepLR(self.optimizer_0, step_size=50, gamma=0.3)
        self.scheduler_1 = optim.lr_scheduler.StepLR(self.optimizer_1, step_size=50, gamma=0.3)


    @torch.no_grad()
    def get_q(self, states, clip, in_grid, sample_latents=None, pair_in=None, pair_out=None, n_samples=1000):
        if sample_latents is not None:
            perm = torch.randperm(self.total_prior_samples)[:n_samples]
            z_samples = torch.FloatTensor(sample_latents).to(self.device).reshape(sample_latents.shape[0]*n_samples,sample_latents.shape[2])
        else:
            z_samples = self.diffusion_prior.sample_extra(states, clip, in_grid, pair_in, pair_out, predict_noise=0, extra_steps=self.extra_steps)

        # q_vals_0 = self.q_net_0(states, z_samples)[:,0]
        # q_vals_1 = self.q_net_1(states, z_samples)[:,0]
        q_vals_0 = self.q_net_0(states, clip, in_grid, z_samples, pair_in, pair_out)[:,0]
        q_vals_1 = self.q_net_1(states, clip, in_grid, z_samples, pair_in, pair_out)[:,0]
        q_vals = torch.minimum(q_vals_0, q_vals_1)
        return z_samples, q_vals


    @torch.no_grad()
    def get_max_skills(self, states, clip, in_grid, pair_in, pair_out, net=0, is_eval=False, sample_latents=None):
        '''
        INPUTS:
            states: batch_size x state_dim
        OUTPUTS:
            max_z: batch_size x z_dim
        '''
        if not is_eval:
            n_states = states.shape[0]
            states = states.repeat_interleave(self.num_prior_samples, 0)
            clip = clip.repeat_interleave(self.num_prior_samples, 0)
            in_grid = in_grid.repeat_interleave(self.num_prior_samples, 0)
            pair_in = pair_in.repeat_interleave(self.num_prior_samples, 0)
            pair_out = pair_out.repeat_interleave(self.num_prior_samples, 0)
            
        if sample_latents is not None:
            perm = torch.randperm(self.total_prior_samples)[:self.num_prior_samples]
            sample_latents = sample_latents[:,perm.cpu().numpy(),:]
            z_samples = torch.FloatTensor(sample_latents).to(self.device).reshape(sample_latents.shape[0]*self.num_prior_samples, sample_latents.shape[2])
        else:
            # z_samples = self.diffusion_prior.sample_extra(states, clip, predict_noise=0, extra_steps=self.extra_steps)
            z_samples = self.diffusion_prior.sample_extra(states, clip, in_grid, pair_in, pair_out, predict_noise=0, extra_steps=self.extra_steps)

        if is_eval:
            q_vals = torch.minimum(self.target_net_0(states, clip, in_grid, z_samples, pair_in, pair_out)[:, 0], self.target_net_1(states, clip, in_grid, z_samples, pair_in, pair_out)[:, 0])
        else:
            if net==0:
                q_vals = self.target_net_0(states, clip, in_grid, z_samples, pair_in, pair_out)[:,0]#self.q_net_0(states,z_samples)[:,0]
            else:
                q_vals = self.target_net_1(states, clip, in_grid, z_samples, pair_in, pair_out)[:,0]#self.q_net_1(states,z_samples)[:,0]

        if is_eval:
            return z_samples, q_vals
        q_vals = q_vals.reshape(n_states, self.num_prior_samples)
        max_vals = torch.max(q_vals, dim=1)
        max_q_vals = max_vals.values
        max_indices = max_vals.indices
        idx = torch.arange(n_states).cuda()*self.num_prior_samples + max_indices 
        max_z = z_samples[idx]

        return max_z, max_q_vals


    def learn(self, dataload_train, dataload_test=None, n_epochs=10000, update_frequency=1, diffusion_model_name='',q_checkpoint_dir = '', cfg_weight=0.0, per_buffer = 0.0, batch_size = 128, gpu_name=None ,task_name='',args=None):
        # assert self.diffusion_prior is not None,
        
        beta = 0.3
        update_steps = 2000
        
        d = datetime.datetime.now()
        task= task_name.split(".")[1]
        os.environ["WANDB_API_KEY"] = "1d8e9524a57e6dc61398747064c13219471115ec"
        
        config = {
                'task':task_name,
                'diffusion_prior':diffusion_model_name,
                'cfg_weight':cfg_weight,
                'per_buffer': per_buffer,
                'beta': beta,
                'update_steps': update_steps,
        }
        
        if args is not None:
            base_config = vars(args).copy()  # args의 모든 속성을 dict로 변환
            config = {**config, **base_config}

        run=wandb.init(
            entity="dbsgh797210",
            project = "LDCQ_single",
            name = 'LDCQ_'+gpu_name+'_'+'Q'+'_'+ task +'_'+str(d.month)+'.'+str(d.day)+'_'+str(d.hour)+'.'+str(d.minute),
            config = config
        )
        print("WandB run initialized with name:", run.name)
        steps_net_0, steps_net_1, steps_total = 0, 0, 0
        self.target_net_0 = copy.deepcopy(self.q_net_0)
        self.target_net_1 = copy.deepcopy(self.q_net_1)
        self.target_net_0.eval()
        self.target_net_1.eval()
        loss_net_0, loss_net_1, loss_total = 0, 0, 0 
        epoch = 0
        update = 0
        
        if not os.path.exists(q_checkpoint_dir) :
            os.makedirs(q_checkpoint_dir)
        for ep in tqdm(range(n_epochs), desc="Epoch"):
            n_batch = 0
            loss_ep = 0
            self.q_net_0.train()
            self.q_net_1.train()
            
            if per_buffer:
                pbar = tqdm(range(len(dataload_train) // batch_size))
                for _ in pbar: # same num_iters as w/o PER
                    # s0, z, reward, sT, dones, indices, weights, max_latents = dataload_train.sample(batch_size, beta)
                    s0, clip0, in_grid, z, reward, sT, clip_T, dones, pair_in, pair_out, indices, weights, max_latents = dataload_train.sample(batch_size, beta)
                    
                    
                    s0 = torch.FloatTensor(s0).to(self.device)
                    clip0 = torch.FloatTensor(clip0).to(self.device)
                    in_grid = torch.FloatTensor(in_grid).to(self.device)
                    z = torch.FloatTensor(z).to(self.device)
                    sT = torch.FloatTensor(sT).to(self.device)
                    clip_T = torch.FloatTensor(clip_T).to(self.device)
                    reward = torch.FloatTensor(reward)[...,None].to(self.device)
                    weights = torch.FloatTensor(weights).to(self.device)
                    dones = torch.FloatTensor(dones).to(self.device)
                    pair_in = torch.FloatTensor(pair_in).to(self.device)
                    pair_out = torch.FloatTensor(pair_out).to(self.device)
                    #net_id = np.random.binomial(n=1, p=0.5, size=(1,))
                    net_id = 0
                    #if net_id==0:
                    self.optimizer_0.zero_grad()

                    q_s0z = self.q_net_0(s0, clip0, in_grid, z, pair_in, pair_out)
                    max_sT_skills,_ = self.get_max_skills(sT, clip_T, in_grid, pair_in, pair_out, net=1-net_id, sample_latents=max_latents)
                    
                    with torch.no_grad():
                        q_sTz = torch.minimum(self.target_net_0(sT, clip_T, in_grid, max_sT_skills.detach(), pair_in, pair_out), self.target_net_1(sT, clip_T, in_grid, max_sT_skills.detach(), pair_in, pair_out),)
                    
                    if 'maze' in diffusion_model_name:
                        q_target = (reward + self.gamma*(reward==0.0)*q_sTz).detach()
                    elif 'kitchen' in diffusion_model_name:
                        q_target = (reward + self.gamma * q_sTz).detach()
                    else:
                        q_target = (reward + (self.gamma**self.horizon)*(dones==0.0)*q_sTz).detach()

                    bellman_loss  = (q_s0z - q_target).pow(2)
                    prios = bellman_loss[...,0] + 5e-6
                    bellman_loss = bellman_loss * weights
                    bellman_loss  = bellman_loss.mean()
                    
                    # bellman_loss = F.mse_loss(q_s0z, q_target)
                    bellman_loss.backward()
                    clip_grad_norm_(self.q_net_0.parameters(), 1)
                    self.optimizer_0.step()
                    loss_net_0 += bellman_loss.detach().item()
                    loss_total += bellman_loss.detach().item()
                    loss_ep += bellman_loss.detach().item()
                    steps_net_0 += 1
                    
                    net_id = 1
                    #else:
                    self.optimizer_1.zero_grad()

                    q_s0z = self.q_net_1(s0, clip0, in_grid, z, pair_in, pair_out)
                    max_sT_skills,_ = self.get_max_skills(sT, clip_T, in_grid, pair_in, pair_out, net=1-net_id, sample_latents=max_latents)

                    with torch.no_grad():
                        q_sTz = torch.minimum(self.target_net_0(sT, clip_T, in_grid, max_sT_skills.detach(), pair_in, pair_out), self.target_net_1(sT, clip_T, in_grid, max_sT_skills.detach(), pair_in, pair_out),)
                    
                    q_target = (reward + (self.gamma**self.horizon)*(dones==0.0)*q_sTz).detach()

                    bellman_loss  = (q_s0z - q_target).pow(2)
                    prios += bellman_loss[...,0] + 5e-6
                    bellman_loss = bellman_loss * weights
                    bellman_loss  = bellman_loss.mean()
                    
                    bellman_loss.backward()
                    clip_grad_norm_(self.q_net_1.parameters(), 1)
                    self.optimizer_1.step()
                    loss_net_1 += bellman_loss.detach().item()
                    loss_total += bellman_loss.detach().item()
                    loss_ep += bellman_loss.detach().item()
                    steps_net_1 += 1

                    dataload_train.update_priorities(indices, prios.data.cpu().numpy()/2)
                    n_batch += 1
                    steps_total += 1
                    pbar.set_description(f"train loss: {loss_ep/n_batch:.4f}")
                    
                    if steps_total%update_frequency == 0:
                        loss_net_0 /= (steps_net_0+1e-4)
                        loss_net_1 /= (steps_net_1+1e-4)
                        loss_total /= 2*update_frequency
                        
                        update += 1
                        wandb.log({"train_Q/train_loss_0": loss_net_0})
                        wandb.log({"train_Q/train_loss_1": loss_net_1})
                        wandb.log({"train_Q/train_loss": loss_total})
                        wandb.log({"train_Q/step_per_update": update/update_steps,"train_Q/steps": steps_total})
                        wandb.log({"train_Q/epoches": ep, "train_Q/steps": steps_total})
                            
                        loss_net_0, loss_net_1, loss_total = 0,0,0
                        steps_net_0, steps_net_1 = 0,0
                        #self.target_net_0 = copy.deepcopy(self.q_net_0)
                        #self.target_net_1 = copy.deepcopy(self.q_net_1)
                        for target_param, local_param in zip(self.target_net_0.parameters(), self.q_net_0.parameters()):
                            target_param.data.copy_((1.0-self.tau)*local_param.data + (self.tau)*target_param.data)
                        for target_param, local_param in zip(self.target_net_1.parameters(), self.q_net_1.parameters()):
                            target_param.data.copy_((1.0-self.tau)*local_param.data + (self.tau)*target_param.data)
                        self.target_net_0.eval()
                        self.target_net_1.eval()
                        
                    if steps_total % (update_steps) == 0:
                        beta = np.min((beta+0.03,1))
                        self.scheduler_0.step()
                        self.scheduler_1.step()
                        
                    if steps_total % (update_steps*10) == 0:
                        torch.save(self,  os.path.join(q_checkpoint_dir,diffusion_model_name+'_dqn_agent_'+str(steps_total//update_steps)+'_cfg_weight_'+str(cfg_weight)+'{}.pt'.format('_PERbuffer' if per_buffer == 1 else '')))
                        
                        # torch.save(self,  parent_folder+'/q_checkpoints/'+diffusion_model_name+'_dqn_agent_'+str(steps_total//update_steps)+'_cfg_weight_'+str(cfg_weight)+'{}.pt'.format('_PERbuffer' if per_buffer == 1 else ''))

                # self.scheduler_0.step()
                # self.scheduler_1.step()
            # experiment.log_metric("train_loss_episode", loss_ep/n_batch, step=epoch)
            wandb.log({"train_Q/train_loss_episode": loss_ep/n_batch, "train_Q/udates": update})
            epoch += 1