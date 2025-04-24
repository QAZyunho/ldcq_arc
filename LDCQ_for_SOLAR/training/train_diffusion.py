import os
import sys
import datetime

curr_folder=os.path.abspath(__file__)
parent_folder=os.path.dirname(os.path.dirname(curr_folder))
sys.path.append(parent_folder) 

from argparse import ArgumentParser
import os
# from comet_ml import Experiment
import wandb

# import d4rl
# import gym
import pickle
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from models.diffusion_models import (
    Model_mlp,
    Model_cnn_mlp,
    Model_Cond_Diffusion,
)


class PriorDataset(Dataset):
    def __init__(
        self, data_dir, filename, train_or_test, test_prop, sample_z=False
    ):
        # just load it all into RAM
        self.state_all = np.load(os.path.join(data_dir, filename + "_states.npy"), allow_pickle=True)
        self.clip_all = np.load(os.path.join(data_dir, filename + "_clip.npy"), allow_pickle=True)
        self.in_grid_all = np.load(os.path.join(data_dir, filename + "_in_grid.npy"), allow_pickle=True)
        self.latent_all = np.load(os.path.join(data_dir, filename + "_latents.npy"), allow_pickle=True)
        if sample_z:
            self.latent_all_std = np.load(os.path.join(data_dir, filename + "_latents_std.npy"), allow_pickle=True)

        # self.state_mean = self.state_all.mean(axis=0)
        # self.state_std = self.state_all.std(axis=0)
        #self.state_all = (self.state_all - self.state_mean) / self.state_std

        # self.latent_mean = self.latent_all.mean(axis=0)
        # self.latent_std = self.latent_all.std(axis=0)
        #self.latent_all = (self.latent_all - self.latent_mean) / self.latent_std
        
        self.sample_z = sample_z
        n_train = int(self.state_all.shape[0] * (1 - test_prop))
        
        if train_or_test == "train":
            self.state_all = self.state_all[:n_train]
            self.clip_all = self.clip_all[:n_train]
            self.in_grid_all = self.in_grid_all[:n_train]
            self.latent_all = self.latent_all[:n_train]
            if sample_z:
                self.latent_all_std = self.latent_all_std[:n_train]
        elif train_or_test == "test":
            self.state_all = self.state_all[n_train:]
            self.clip_all = self.clip_all[n_train:]
            self.in_grid_all = self.in_grid_all[n_train:]
            self.latent_all = self.latent_all[n_train:]
            if sample_z:
                self.latent_all_std = self.latent_all_std[n_train:]
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
            # latent = (latent - self.latent_mean) / self.latent_std
        #else:
        #    latent = (latent - self.latent_mean) / self.latent_std
        return (state, clip, in_grid, latent)


def train(args):
    # get datasets set up
    torch_data_train = PriorDataset(
        args.data_dir, args.skill_model_filename[:-4], train_or_test="train", test_prop=args.test_split, sample_z=args.sample_z
    )
    dataload_train = DataLoader(
        torch_data_train, batch_size=args.batch_size, shuffle=True, num_workers=0
    )

    if args.test_split > 0.0:
        torch_data_test = PriorDataset(
            args.data_dir, args.skill_model_filename[:-4], train_or_test="test", test_prop=args.test_split, sample_z=args.sample_z
        )
        dataload_test = DataLoader(
            torch_data_test, batch_size=args.batch_size, shuffle=True, num_workers=0
        )

    # x_shape = torch_data_train.state_all.shape[1]
    x_shape = args.s_dim
    y_dim = torch_data_train.latent_all.shape[1]

    # create model
    nn_model = Model_mlp(
        x_shape = x_shape, 
        n_hidden = args.n_hidden, 
        y_dim = y_dim, 
        embed_dim = 128,    # h_dim*8 = 16*8 = 128
        net_type = args.net_type,
        max_grid_size=args.max_grid_size
    ).to(args.device)
    
    model = Model_Cond_Diffusion(
        nn_model,
        betas=(1e-4, 0.02),
        n_T=args.diffusion_steps,
        device=args.device,
        x_dim=x_shape,
        y_dim=y_dim,
        drop_prob=args.drop_prob,
        guide_w=0.0,
        # normalize_latent=args.normalize_latent,   # 여기도 normalize 키면 학습이 안됨
        schedule=args.schedule,
    ).to(args.device)

    # Select Optimizer
    if(args.optimizer == "Adam"):
        optim = torch.optim.Adam(model.parameters(), lr=args.lrate)
    elif(args.optimizer == "AdamW"):
        optim = torch.optim.AdamW(model.parameters(), lr=args.lrate)
    else:
        ValueError("올바른 Optimizer를 입력해주세요")
    
    
    best_test_loss = 10000000

    for ep in tqdm(range(args.n_epoch), desc="Epoch"):
        model.train()

        # lrate decay
        #optim.param_groups[0]["lr"] = args.lrate * ((np.cos((ep / args.n_epoch) * np.pi) + 1) / 2)
        optim.param_groups[0]["lr"] = args.lrate * ((np.cos((ep / 75) * np.pi) + 1))

        # train loop
        model.train()
        pbar = tqdm(dataload_train)
        loss_ep, n_batch = 0, 0

        for x_batch, clip_batch, in_grid_batch, y_batch in pbar:
            x_batch = x_batch.type(torch.FloatTensor).to(args.device)   # (Batch, 1, 30, 30)
            clip_batch = clip_batch.type(torch.FloatTensor).to(args.device)   # (Batch, 1, 30, 30)
            in_grid_batch = in_grid_batch.type(torch.FloatTensor).to(args.device)   # (Batch, 1, 30, 30)
            y_batch = y_batch.type(torch.FloatTensor).to(args.device)   # (Batch, z_dim)
            
            loss = model.loss_on_batch(x_batch, clip_batch, in_grid_batch, y_batch, args.predict_noise)
            optim.zero_grad()
            loss.backward()
            wandb.log({"train_diffusion/loss": loss.item()})
            loss_ep += loss.detach().item()
            n_batch += 1
            pbar.set_description(f"train loss: {loss_ep/n_batch:.4f}")
            optim.step()
        # experiment.log_metric("train_loss", loss_ep/n_batch, step=ep)
        wandb.log({"train_diffusion/mean_loss": loss_ep/n_batch})
        torch.save(nn_model, os.path.join(args.checkpoint_dir, args.skill_model_filename[:-4] + '_diffusion_prior.pt'))

        # test loop
        if args.test_split > 0.0:
            model.eval()
            pbar = tqdm(dataload_test)
            loss_ep, n_batch = 0, 0

            with torch.no_grad():
                for x_batch, clip_batch, in_grid_batch, y_batch in pbar:
                    x_batch = x_batch.type(torch.FloatTensor).to(args.device)
                    clip_batch = clip_batch.type(torch.FloatTensor).to(args.device)
                    in_grid_batch = in_grid_batch.type(torch.FloatTensor).to(args.device)
                    y_batch = y_batch.type(torch.FloatTensor).to(args.device)
                    
                    loss = model.loss_on_batch(x_batch, clip_batch, in_grid_batch, y_batch, args.predict_noise)
                    loss_ep += loss.detach().item()
                    n_batch += 1
                    pbar.set_description(f"test loss: {loss_ep/n_batch:.4f}")
            # experiment.log_metric("test_loss", loss_ep/n_batch, step=ep)
            wandb.log({"train_diffusion/test_loss": loss_ep/n_batch})

            if loss_ep < best_test_loss:
                best_test_loss = loss_ep
                torch.save(nn_model, os.path.join(args.checkpoint_dir, args.skill_model_filename[:-4] + '_diffusion_prior_best.pt'))

        # elif ep%75==0:
        #     torch.save(nn_model, os.path.join(args.checkpoint_dir, args.skill_model_filename[:-4] + '_diffusion_prior_best.pt'))

        if(ep%args.save_cycle == 0):
            torch.save(nn_model, os.path.join(args.checkpoint_dir, args.skill_model_filename[:-4]+'_'+str(ep)+'_epoch'+'.pth'))

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('--env', type=str, default='antmaze-large-diverse-v2')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--n_epoch', type=int, default=100)
    parser.add_argument('--lrate', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--net_type', type=str, default='unet')
    parser.add_argument('--n_hidden', type=int, default=512)
    parser.add_argument('--test_split', type=float, default=0.05)    # 원래는 0.1
    parser.add_argument('--sample_z', type=int, default=0)

    parser.add_argument('--solar_dir', type=str, default=None)
    parser.add_argument('--checkpoint_dir', type=str, default=parent_folder+'/checkpoints/')
    parser.add_argument('--data_dir', type=str, default=parent_folder+'/data/')
    parser.add_argument('--skill_model_filename', type=str)
    parser.add_argument('--append_goals', type=int, default=0)

    parser.add_argument('--drop_prob', type=float, default=0.1)     # 원래 코드는 0
    parser.add_argument('--diffusion_steps', type=int, default=100)
    parser.add_argument('--cfg_weight', type=float, default=0.0)
    parser.add_argument('--predict_noise', type=int, default=0)
    parser.add_argument('--normalize_latent', type=int, default=1)  # 원래는 0(바활성화)
    parser.add_argument('--schedule', type=str, default='linear')

    # parser.add_argument('--a_dim', type=int, default=36)
    # parser.add_argument('--z_dim', type=int, default=128)            # 원래는 16
    # parser.add_argument('--h_dim', type=int, default=256)
    parser.add_argument('--s_dim', type=int, default=256)
    parser.add_argument('--date', type=str, default='00.00')

    parser.add_argument('--save_cycle', type=int, default=100)
    parser.add_argument('--gpu_name', type=str, required=True)
    parser.add_argument('--optimizer', type=str, default="AdamW")
    parser.add_argument('--max_grid_size', type=int, default=30)
    args = parser.parse_args()

    d = datetime.datetime.now()
    file_info = args.env+'_'+args.date
    filename = args.gpu_name+'_'+'diffusion_'+file_info
    task_name = args.solar_dir.split("/")[-1]
    task= task_name.split(".")[1]
    wandb.init(
        project = "LDCQ_single",
        name = 'LDCQ_'+args.gpu_name+'_'+'diffusion'+'_'+ task + '_'+ str(d.month)+'.'+str(d.day)+'_'+str(d.hour)+'.'+str(d.minute),
        config = {
            'task':task_name,
            'lr':args.lrate,
            'batch_size':args.batch_size,
            'sample_z':args.sample_z,
            'env_name':args.env,
            'filename':filename,
            'net_type':args.net_type,
            'diffusion_steps':args.diffusion_steps,
            'skill_model_filename':args.skill_model_filename,
            'normalize_latent':args.normalize_latent,
            'schedule': args.schedule,
            'test_split': args.test_split,
            'append_goals': args.append_goals
        }
    )
    
    train(args)
