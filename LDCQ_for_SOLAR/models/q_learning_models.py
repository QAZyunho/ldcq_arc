import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random

class MLP_Q(nn.Module):
    '''
    Q(s,z) is our Abstract MLP Q function which takes as input current state s and skill z, 
    and outputs the expected return on executing the skill
    '''
    def __init__(self, state_dim, z_dim, h_dim=256,max_grid_size=30):
        super(MLP_Q,self).__init__()

        self.h_dim = h_dim
        self.z_dim = z_dim
        self.max_grid_size = max_grid_size
        # z_embed_dim = h_dim//2
        z_embed_dim = h_dim
        self.state_mlp = nn.Sequential(
            nn.Linear(state_dim, h_dim),
            nn.LayerNorm(h_dim),
            nn.GELU(),
            nn.Linear(h_dim, h_dim),
            nn.LayerNorm(h_dim),
            nn.GELU()
        )
        self.clip_mlp = nn.Sequential(
            nn.Linear(state_dim, h_dim),
            nn.LayerNorm(h_dim),
            nn.GELU(),
            nn.Linear(h_dim, h_dim),
            nn.LayerNorm(h_dim),
            nn.GELU()
        )
        self.in_grid_mlp = nn.Sequential(
            nn.Linear(state_dim, h_dim),
            nn.LayerNorm(h_dim),
            nn.GELU(),
            nn.Linear(h_dim, h_dim),
            nn.LayerNorm(h_dim),
            nn.GELU()
        )
        self.latent_mlp = nn.Sequential(
            nn.Linear(z_dim, z_embed_dim),
            nn.LayerNorm(z_embed_dim),
            nn.GELU(),
            nn.Linear(z_embed_dim, z_embed_dim),
            nn.LayerNorm(z_embed_dim),
            nn.GELU()
        )
        self.output_mlp = nn.Sequential(
            nn.Linear(h_dim+h_dim+h_dim+z_embed_dim+h_dim, 128),    # state, clip, in_grid, z, pair
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, 1)
        )
        self.state_emb_layer = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=1, stride=1, padding=0), 
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0), 
            # nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0), 
            nn.ReLU(),
            nn.Flatten(), nn.Linear(32*self.max_grid_size*self.max_grid_size, h_dim), 
            # nn.Flatten(), nn.Linear(32*28*28, h_dim), 
            nn.Tanh()
        )
        self.pair_emb_layer = nn.Sequential(
            nn.Linear(6*h_dim, h_dim),  # input-output pair 총 3개씩
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU()
        )
        
    def forward(self, s, clip, in_grid, z, pair_in, pair_out):
        '''
        INPUTS:
            s   : batch_size x 1 x self.max_grid_size x self.max_grid_size
            clip: batch_size x 1 x self.max_grid_size x self.max_grid_size
            z   : batch_size x z_dim
            pair_in: batch_size x 3 x self.max_grid_size x self.max_grid_size
            pair_out: batch_size x 3 x self.max_grid_size x self.max_grid_size
        OUTPUS:
            q_sz: batch_size x 1
        '''
        # 2차원 ARC 그리드를 1차원으로 임베딩
        s_emb = self.state_emb_layer(s)
        clip_emb = self.state_emb_layer(clip)
        in_grid_emb = self.state_emb_layer(in_grid)

        pair = torch.cat([pair_in, pair_out], dim=1)
        
        pair_shape = pair.shape
        pair_emb = self.state_emb_layer(pair.reshape(-1, 1, self.max_grid_size, self.max_grid_size).type(torch.float32).contiguous())
        pair_emb = pair_emb.reshape(pair_shape[0], 1, pair_shape[1]*self.h_dim)
        pair_embed = self.pair_emb_layer(pair_emb).squeeze(1)
        
        state_embed = self.state_mlp(s_emb)
        clip_embed = self.clip_mlp(clip_emb)
        in_grid_embed = self.in_grid_mlp(in_grid_emb)
        z_embed = self.latent_mlp(z)

        s_z_cat = torch.cat([state_embed, clip_embed, in_grid_embed, z_embed, pair_embed], dim=1)
        q_sz = self.output_mlp(s_z_cat)
        return q_sz