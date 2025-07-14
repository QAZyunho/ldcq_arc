import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader
from torch.distributions.transformed_distribution import TransformedDistribution
import torch.distributions.normal as Normal
import torch.distributions.kl as KL
from utils.utils import reparameterize
from models.diffusion_models import (
    Model_mlp,
    Model_cnn_mlp,
    Model_Cond_Diffusion,
)

class AbstractDynamics(nn.Module):
    '''
    P(s_T|s_0,z) is our "abstract dynamics model", because it predicts the resulting state transition over T timesteps given a skill 
    (so similar to regular dynamics model, but in skill space and also temporally extended)
    See Encoder and Decoder for more description
    '''
    def __init__(self,state_dim,z_dim,h_dim,per_element_sigma=True):

        super(AbstractDynamics,self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(state_dim+z_dim,h_dim),
            nn.ReLU(),
            nn.Linear(h_dim,h_dim),
            nn.ReLU())
        self.mean_layer = nn.Sequential(
            nn.Linear(h_dim,h_dim),
            nn.ReLU(),
            nn.Linear(h_dim,state_dim))
        if per_element_sigma:
            self.sig_layer  = nn.Sequential(nn.Linear(h_dim,h_dim),nn.ReLU(),nn.Linear(h_dim,state_dim),nn.Softplus())
        else:
            self.sig_layer = nn.Sequential(nn.Linear(h_dim,h_dim),nn.ReLU(),nn.Linear(h_dim,1),nn.Softplus())

        self.state_dim = state_dim
        self.per_element_sigma = per_element_sigma

    def forward(self,s0,z):

        '''
        INPUTS:
            s0: batch_size x 1 x state_dim initial state (first state in execution of skill)
            z:  batch_size x 1 x z_dim "skill"/z
        OUTPUTS: 
            sT_mean: batch_size x 1 x state_dim tensor of terminal (time=T) state means
            sT_sig:  batch_size x 1 x state_dim tensor of terminal (time=T) state standard devs
        '''

        # concatenate s0 and z
        s0_z = torch.cat([s0,z],dim=-1)
        # pass s0_z through layers
        feats = self.layers(s0_z)
        # get mean and stand dev of action distribution
        sT_mean = self.mean_layer(feats)
        sT_sig  = self.sig_layer(feats)

        if not self.per_element_sigma:
            sT_sig = torch.cat(self.state_dim*[sT_sig],dim=-1)

        return sT_mean, sT_sig


class AutoregressiveStateDecoder(nn.Module):
    '''
    P(s_T|s_0,z) is our "low-level policy", so this is the feedback policy the agent runs while executing skill described by z.
    See Encoder and Decoder for more description
    '''
    def __init__(self,state_dim,z_dim,h_dim,per_element_sigma=True):

        super(AutoregressiveStateDecoder,self).__init__()
        self.decoder_components = nn.ModuleList([LowLevelPolicy(state_dim+i,1,z_dim,h_dim,a_dist='normal') for i in range(state_dim)])
        self.state_dim = state_dim

    def forward(self,state,s_T,z, evaluation=False):
        '''
        INPUTS:
            state: batch_size x 1 x state_dim tensor of states 
            action: batch_size x 1 x a_dim tensor of actions
            z:     batch_size x 1 x z_dim tensor of states
        OUTPUTS:
            a_mean: batch_size x T x a_dim tensor of action means for each t in {0.,,,.T}
            a_sig:  batch_size x T x a_dim tensor of action standard devs for each t in {0.,,,.T}
        
        Iterate through each low level policy component.
        The ith element gets to condition on all elements up to but NOT including a_i
        '''
        s_means = []
        s_sigs = []

        s_means_tensor = torch.zeros_like(state)
        s_sigs_tensor = torch.zeros_like(state)

        for i in range(self.state_dim):
            # Concat state, and a up to i.  state_a takes place of state in orginary policy.
            if not evaluation:
                state_a = torch.cat([state, s_T[:,:,:i]],dim=-1)
            else:
                state_a = torch.cat([state, s_means_tensor[:, :, :i].detach()], dim=-1)
            # pass through ith policy component
            s_T_mean_i,s_T_sig_i = self.decoder_components[i](state_a,z) # these are batch_size x T x 1
            # add to growing list of policy elements
            s_means.append(s_T_mean_i)
            s_sigs.append(s_T_sig_i)

            if evaluation:
                s_means_tensor = torch.cat(s_means, dim=-1)
                s_sigs_tensor = torch.cat(s_sigs, dim=-1)

        s_means = torch.cat(s_means,dim=-1)
        s_sigs  = torch.cat(s_sigs, dim=-1)
        return s_means, s_sigs
    
    def sample(self,state,z):
        states = []
        for i in range(self.state_dim):
            # Concat state, a up to i, and z_tiled
            state_a = torch.cat([state]+states,dim=-1)
            # pass through ith policy component
            s_T_mean_i,s_T_sig_i = self.decoder_components[i](state_a,z)  # these are batch_size x T x 1
            s_i = reparameterize(s_T_mean_i,s_T_sig_i)
            states.append(s_i)

        return torch.cat(states,dim=-1)

    
    def numpy_dynamics(self,state,z):
        '''
        maps state as a numpy array and z as a pytorch tensor to a numpy action
        '''
        state = torch.reshape(torch.tensor(state,device=torch.device('cuda:0'),dtype=torch.float32),(1,1,-1))
        
        s_T = self.sample(state,z)
        s_T = s_T.detach().cpu().numpy()
        
        return s_T.reshape([self.state_dim,])


class LowLevelPolicy(nn.Module):
    '''
    P(a_t|s_t,z) is our "low-level policy", so this is the feedback policy the agent runs while executing skill described by z.
    See Encoder and Decoder for more description
    '''
    def __init__(self,state_dim,a_dim,z_dim,h_dim,a_dist,action_num,max_grid_size,fixed_sig=None,state_emb_layer=None,pair_emb_layer=None):

        super(LowLevelPolicy, self).__init__()

        self.state_emb_layer = state_emb_layer
        self.pair_emb_layer = pair_emb_layer
        
        self.layers = nn.Sequential(
            nn.Linear(h_dim+h_dim+h_dim+z_dim+h_dim, h_dim),  # state, clip, in_grid, z, pair
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU()
        )
        self.h_dim = h_dim
        self.max_grid_size = max_grid_size
        
        self.a_layer = nn.Sequential(nn.Linear(h_dim,h_dim),nn.ReLU(),nn.Linear(h_dim,action_num))
        self.a_act = nn.Softmax(dim=2)
        
        self.x_layer = nn.Sequential(nn.Linear(h_dim,h_dim),nn.ReLU(),nn.Linear(h_dim,self.max_grid_size))
        self.x_act = nn.Softmax(dim=2)
        
        self.y_layer = nn.Sequential(nn.Linear(h_dim,h_dim),nn.ReLU(),nn.Linear(h_dim,self.max_grid_size))
        self.y_act = nn.Softmax(dim=2)
        
        self.h_layer = nn.Sequential(nn.Linear(h_dim,h_dim),nn.ReLU(),nn.Linear(h_dim,self.max_grid_size))
        self.h_act = nn.Softmax(dim=2)
        
        self.w_layer = nn.Sequential(nn.Linear(h_dim,h_dim),nn.ReLU(),nn.Linear(h_dim,self.max_grid_size))
        self.w_act = nn.Softmax(dim=2)
        
        self.a_dist = a_dist
        self.a_dim = a_dim
        self.fixed_sig = fixed_sig

    def forward(self, state, clip, in_grid, z, pair_in, pair_out):
        '''
        INPUTS:
            state: batch_size x T x state_dim tensor of states 
            z:     batch_size x 1 x z_dim tensor of states
        OUTPUTS:
            a_mean: batch_size x T x a_dim tensor of action means for each t in {0.,,,.T}
            a_sig:  batch_size x T x a_dim tensor of action standard devs for each t in {0.,,,.T}
        '''
        # tile z along time axis so dimension matches state
        # z_tiled = z.tile([1, state.shape[-2], 1]) #not sure about this// state에 붙이려고 batch_size x T x state_dim형태로 변환

        # 원본 Concat state and z_tiled
        # state_z = torch.cat([state, z_tiled], dim=-1)
        
        # ARC 전용 Concat state and z_tiled
        s_emb = self.state_emb_layer(state.reshape(-1, 1, self.max_grid_size, self.max_grid_size).type(torch.float32).contiguous())    # (batch * block_size, h_dim)
        s_emb = s_emb.reshape(state.shape[0], state.shape[1], self.h_dim) # (batch, block_size, n_embd)
        
        # clip 임베딩
        clip_emb = self.state_emb_layer(clip.reshape(-1, 1, self.max_grid_size, self.max_grid_size).type(torch.float32).contiguous())    # (batch * block_size, h_dim)
        clip_emb = clip_emb.reshape(clip.shape[0], clip.shape[1], self.h_dim) # (batch, block_size, n_embd)
        
        # in_grid 임베딩
        in_grid_emb = self.state_emb_layer(in_grid.reshape(-1, 1, self.max_grid_size, self.max_grid_size).type(torch.float32).contiguous())
        in_grid_emb = in_grid_emb.reshape(in_grid.shape[0], in_grid.shape[1], self.h_dim)
        in_grid_tiled = in_grid_emb.tile([1, s_emb.shape[-2], 1])       # 차원 맞춰주려고 N번 반복
        
        # z 반복 - 차원 맞춰주려고 N번 반복
        z_tiled = z.tile([1, s_emb.shape[-2], 1])

        # input-output pair 임베딩
        pair = torch.cat([pair_in, pair_out], dim=1)
        pair_shape = pair.shape
        # print("LL policy - pair shape : {0}".format(pair.shape))
        
        pair_emb = self.state_emb_layer(pair.reshape(-1, 1, self.max_grid_size, self.max_grid_size).type(torch.float32).contiguous())
        # print("LL policy - Before pair_emb : {0}".format(pair_emb.shape))

        pair_emb = pair_emb.reshape(pair_shape[0], 1, pair_shape[1]*self.h_dim)
        # print("LL policy - After pair_emb : {0}".format(pair_emb.shape))
        
        pair_emb = self.pair_emb_layer(pair_emb)
        pair_tiled = pair_emb.tile([1, s_emb.shape[-2], 1])       # 차원 맞춰주려고 N번 반복
        
        state_z = torch.cat([s_emb, clip_emb, in_grid_tiled, z_tiled, pair_tiled], dim=-1)

        # pass z and state through layers
        feats = self.layers(state_z)
        
        # get mean and stand dev of action distribution
        a_mean = self.a_layer(feats)
        a_mean = self.a_act(a_mean)
        
        x_mean = self.x_layer(feats)
        x_mean = self.x_act(x_mean)
        
        y_mean = self.y_layer(feats)
        y_mean = self.y_act(y_mean)
        
        h_mean = self.h_layer(feats)
        h_mean = self.h_act(h_mean)
        
        w_mean = self.w_layer(feats)
        w_mean = self.w_act(w_mean)
        
        return a_mean, None, x_mean, None, y_mean, None, h_mean, None, w_mean, None
    
    def numpy_policy(self, state, clip, in_grid, z, pair_in, pair_out):
        '''
        maps state as a numpy array and z as a pytorch tensor to a numpy action
        '''
        state = torch.reshape(state.clone().detach().to(device=torch.device('cuda:0'), dtype=torch.float32), (1,1,-1))
        clip = torch.reshape(clip.clone().detach().to(device=torch.device('cuda:0'), dtype=torch.float32), (1,1,-1))
        in_grid = torch.reshape(in_grid.clone().detach().to(device=torch.device('cuda:0'), dtype=torch.float32), (1,1,-1))

        # state = torch.reshape(torch.tensor(state, device=torch.device('cuda:0'), dtype=torch.float32), (1,1,-1))
        # clip = torch.reshape(torch.tensor(clip, device=torch.device('cuda:0'), dtype=torch.float32), (1,1,-1))
        # in_grid = torch.reshape(torch.tensor(in_grid, device=torch.device('cuda:0'), dtype=torch.float32), (1,1,-1))
        
        a_mean, a_sig, x_mean, x_sig, y_mean, y_sig, h_mean, h_sig, w_mean, w_sig = self.forward(state, clip, in_grid, z, pair_in, pair_out)
        # a_mean, a_sig = self.forward(state, z)
        action = self.reparameterize(a_mean, a_sig, self.a_dim)
        
        x = self.reparameterize(x_mean, x_sig, self.max_grid_size)
        y = self.reparameterize(y_mean, y_sig, self.max_grid_size)
        h = self.reparameterize(h_mean, h_sig, self.max_grid_size)
        w = self.reparameterize(w_mean, w_sig, self.max_grid_size)
        
        if self.a_dist == 'tanh_normal':
            action = nn.Tanh()(action)
            x = nn.Tanh()(x)
            y = nn.Tanh()(y)
            h = nn.Tanh()(h)
            w = nn.Tanh()(w)
            
        action = action.detach().cpu().numpy()
        x = x.detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        h = h.detach().cpu().numpy()
        w = w.detach().cpu().numpy()
        
        return action, x, y, h, w
        # return action.reshape([self.a_dim,])
        
    def tensor_policy(self, state, clip, in_grid, z, pair_in, pair_out):
        '''
        maps state as a numpy array and z as a pytorch tensor to a numpy action
        '''
        state_shape = state.shape
        state = torch.reshape(torch.tensor(state.clone().detach(), device=torch.device('cuda:0'), dtype=torch.float32), (state_shape[0],1,-1))
        
        clip_shape = clip.shape
        clip = torch.reshape(torch.tensor(clip.clone().detach(), device=torch.device('cuda:0'), dtype=torch.float32), (clip_shape[0],1,-1))
        
        in_grid_shape = in_grid.shape
        in_grid = torch.reshape(torch.tensor(in_grid.clone().detach(), device=torch.device('cuda:0'), dtype=torch.float32), (in_grid_shape[0],1,-1))
        
        a_mean, a_sig, x_mean, x_sig, y_mean, y_sig, h_mean, h_sig, w_mean, w_sig = self.forward(state, clip, in_grid, z, pair_in, pair_out)
        # a_mean, a_sig = self.forward(state, z)
        action = self.reparameterize(a_mean, a_sig, self.a_dim)
        
        x = self.reparameterize(x_mean, x_sig, self.max_grid_size)
        y = self.reparameterize(y_mean, y_sig, self.max_grid_size)
        h = self.reparameterize(h_mean, h_sig, self.max_grid_size)
        w = self.reparameterize(w_mean, w_sig, self.max_grid_size)
        
        if self.a_dist == 'tanh_normal':
            action = nn.Tanh()(action)
            x = nn.Tanh()(x)
            y = nn.Tanh()(y)
            h = nn.Tanh()(h)
            w = nn.Tanh()(w)
        
        return action, x, y, h, w
    
    def reparameterize(self, mean, std, dim):
        if self.a_dist=='softmax':
            intervals = torch.linspace(-1, 1, dim).cuda()
            max_idx = torch.argmax(mean, dim=2).unsqueeze(2)
            max_interval = intervals[max_idx]
            return max_interval
        
        if(std == None):
            return mean
        else:
            eps = torch.normal(torch.zeros(mean.size()).cuda(), torch.ones(mean.size()).cuda())
            return mean + std*eps


class AutoregressiveLowLevelPolicy(nn.Module):
    '''
    P(a_t|s_t,z) is our "low-level policy", so this is the feedback policy the agent runs while executing skill described by z.
    See Encoder and Decoder for more description
    '''
    def __init__(self,state_dim,a_dim,z_dim,h_dim,a_dist,fixed_sig=None):

        super(AutoregressiveLowLevelPolicy,self).__init__()
        self.policy_components = nn.ModuleList([LowLevelPolicy(state_dim+i,1,z_dim,h_dim,a_dist=a_dist,fixed_sig=fixed_sig) for i in range(a_dim)])
        self.a_dim = a_dim
        self.a_dist = a_dist
        
    def forward(self,state,actions,z):
        '''
        INPUTS:
            state: batch_size x T x state_dim tensor of states
            action: batch_size x T x a_dim tensor of actions
            z:     batch_size x 1 x z_dim tensor of states
        OUTPUTS:
            a_mean: batch_size x T x a_dim tensor of action means for each t in {0.,,,.T}
            a_sig:  batch_size x T x a_dim tensor of action standard devs for each t in {0.,,,.T}
        
        Iterate through each low level policy component.
        The ith element gets to condition on all elements up to but NOT including a_i
        '''
        a_means = []
        a_sigs = []
        for i in range(self.a_dim):
            # Concat state, and a up to i.  state_a takes place of state in orginary policy.
            state_a = torch.cat([state,actions[:,:,:i]],dim=-1)
            # pass through ith policy component
            a_mean_i,a_sig_i = self.policy_components[i](state_a,z)  # these are batch_size x T x 1
            if self.a_dist == 'softmax':
                a_mean_i = a_mean_i.unsqueeze(dim=2)
            # add to growing list of policy elements
            a_means.append(a_mean_i)
            if not self.a_dist == 'softmax':
                a_sigs.append(a_sig_i)
        if self.a_dist == 'softmax':
            a_means = torch.cat(a_means,dim=2)
            return a_means, None
        a_means = torch.cat(a_means,dim=-1)
        a_sigs  = torch.cat(a_sigs, dim=-1)
        return a_means, a_sigs
    
    def sample(self,state,z):
        actions = []
        for i in range(self.a_dim):
            # Concat state, a up to i, and z_tiled
            state_a = torch.cat([state]+actions,dim=-1)
            # pass through ith policy component
            a_mean_i,a_sig_i = self.policy_components[i](state_a,z)  # these are batch_size x T x 1

            a_i = self.reparameterize(a_mean_i,a_sig_i)
            #a_i = a_mean_i

            if self.a_dist == 'tanh_normal':
                a_i = nn.Tanh()(a_i)
            actions.append(a_i)

        return torch.cat(actions,dim=-1)
    
    def numpy_policy(self,state,z):
        '''
        maps state as a numpy array and z as a pytorch tensor to a numpy action
        '''
        state = torch.reshape(torch.tensor(state,device=torch.device('cuda:0'),dtype=torch.float32),(1,1,-1))
        
        action = self.sample(state,z)
        action = action.detach().cpu().numpy()
        
        return action.reshape([self.a_dim,])

    def reparameterize(self, mean, std):
        if self.a_dist=='softmax':
            intervals = torch.linspace(-1, 1, 21).cuda()
            # max_idx = torch.distributions.categorical.Categorical(mean).sample()
            max_idx = torch.argmax(mean, dim=2)
            max_interval = intervals[max_idx]
            return max_interval.unsqueeze(-1)
        eps = torch.normal(torch.zeros(mean.size()).cuda(), torch.ones(mean.size()).cuda())
        return mean + std*eps


class TransformEncoder(nn.Module):
    '''
    Encoder module.
    -Concat states+actions
    -Pass through linear embedding
    -Pass through bidirectional RNN
    -Pass output of bidirectional RNN through 2 linear layers, one to get mean of z and one to get stand dev (we're estimating one z ("skill") for entire episode)
    '''
    def __init__(self,state_dim,a_dim,z_dim,h_dim, horizon,
                          n_layers=3, n_heads=4, dropout = 0.1):
        super(TransformEncoder, self).__init__()

        self.horizon = horizon
        self.embed_state = torch.nn.Linear(state_dim, h_dim)
        self.embed_action = torch.nn.Linear(a_dim, h_dim)
        self.embed_ln = nn.LayerNorm(h_dim)
        
        # Last token is special -> used for z prediction
        self.embed_timestep = nn.Embedding(horizon+1, h_dim)

        encoder_layer = nn.TransformerEncoderLayer(h_dim, nhead = n_heads,
                                       dim_feedforward=4*h_dim, dropout=dropout)
        self.transformer_model = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.mean_layer = nn.Sequential(nn.Linear(h_dim,h_dim),nn.ReLU(),nn.Linear(h_dim,z_dim))

        self.sig_layer  = nn.Sequential(nn.Linear(h_dim,h_dim),nn.ReLU(),nn.Linear(h_dim,z_dim),nn.Softplus())


    def forward(self,states,actions):

        '''
        Takes a sequence of states and actions, and infers the distribution over latent skill variable, z
        
        INPUTS:
            states: batch_size x T x state_dim state sequence tensor
            actions: batch_size x T x a_dim action sequence tensor
        OUTPUTS:
            z_mean: batch_size x 1 x z_dim tensor indicating mean of z distribution
            z_sig:  batch_size x 1 x z_dim tensor indicating standard deviation of z distribution
        '''
        timesteps = self.embed_timestep(torch.arange(actions.shape[1]).to(actions.device))
        timesteps = timesteps.unsqueeze(0).repeat((actions.shape[0], 1, 1))
        z_embedding = self.embed_timestep(torch.LongTensor([self.horizon]).to(actions.device))
        z_embedding = z_embedding.unsqueeze(0).repeat((actions.shape[0], 1, 1))
  
        state_latent = self.embed_state(states) + timesteps
        action_latent = self.embed_action(actions) + timesteps
        
        transformer_inputs = torch.cat([state_latent, action_latent, z_embedding], dim = 1)
        transformer_inputs = self.embed_ln(transformer_inputs)

        transformer_outputs = self.transformer_model(transformer_inputs)

        hn = transformer_outputs[:, -1]

        z_mean = self.mean_layer(hn).unsqueeze(1)
        z_sig = self.sig_layer(hn).unsqueeze(1)
        
        return z_mean, z_sig


class GRUEncoder(nn.Module):
    '''
    Encoder module.
    -Concat states+actions
    -Pass through linear embedding
    -Pass through bidirectional RNN
    -Pass output of bidirectional RNN through 2 linear layers, one to get mean of z and one to get stand dev (we're estimating one z ("skill") for entire episode)
    '''
    def __init__(self,state_dim,a_dim,z_dim,h_dim,n_gru_layers=4,normalize_latent=False,
                 color_num=11,action_num=36,max_grid_size=30,state_emb_layer=None,pair_emb_layer=None):
        super(GRUEncoder, self).__init__()

        self.state_dim = state_dim # state dimension
        self.a_dim = a_dim # action dimension
        self.normalize_latent = normalize_latent
        self.h_dim = h_dim
        self.max_grid_size = max_grid_size
        # 아래 방법은 1차원으로 펴서 사용하는 방법
        # self.state_emb_layer  = nn.Sequential(
        #     nn.Embedding(color_num, h_dim),
        #     nn.ReLU(),
        #     nn.Linear(h_dim, h_dim),
        #     nn.ReLU()
        # )

        self.state_emb_layer = state_emb_layer
        self.pair_emb_layer = pair_emb_layer

        self.action_emb_layer  = nn.Sequential(nn.Embedding(action_num, h_dim),nn.ReLU(),nn.Linear(h_dim, h_dim),nn.ReLU())
        self.x_emb_layer  = nn.Sequential(nn.Embedding(self.max_grid_size, h_dim),nn.ReLU(),nn.Linear(h_dim, h_dim),nn.ReLU())
        self.y_emb_layer  = nn.Sequential(nn.Embedding(self.max_grid_size, h_dim),nn.ReLU(),nn.Linear(h_dim, h_dim),nn.ReLU())
        self.h_emb_layer  = nn.Sequential(nn.Embedding(self.max_grid_size, h_dim),nn.ReLU(),nn.Linear(h_dim, h_dim),nn.ReLU())
        self.w_emb_layer  = nn.Sequential(nn.Embedding(self.max_grid_size, h_dim),nn.ReLU(),nn.Linear(h_dim, h_dim),nn.ReLU())
        
        # self.rnn = nn.GRU(h_dim+a_dim, h_dim, batch_first=True, bidirectional=True, num_layers=n_gru_layers)
        self.rnn = nn.GRU(h_dim, h_dim, batch_first=True, bidirectional=True, num_layers=n_gru_layers)

        #self.mean_layer = nn.Linear(h_dim,z_dim)
        self.mean_layer = nn.Sequential(
            nn.Linear(2*h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, z_dim)
        )
        #self.sig_layer  = nn.Sequential(nn.Linear(h_dim,z_dim),nn.Softplus())  # using softplus to ensure stand dev is positive
        self.sig_layer  = nn.Sequential(
            nn.Linear(2*h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, z_dim),
            nn.Softplus()
        )

    def forward(self, states, clip, in_grid, actions, selection, pair_in, pair_out):

        '''
        Takes a sequence of states and actions, and infers the distribution over latent skill variable, z
        
        INPUTS:
            states: batch_size x T x state_dim state sequence tensor                   
            actions: batch_size x T x a_dim action sequence tensor
        OUTPUTS:
            z_mean: batch_size x 1 x z_dim tensor indicating mean of z distribution
            z_sig:  batch_size x 1 x z_dim tensor indicating standard deviation of z distribution
        '''
        # State가 1차원인 경우
        # s_emb = self.state_emb_layer(states)
        
        # State가 2차원인 경우
        s_emb = self.state_emb_layer(states.reshape(-1, 1, self.max_grid_size, self.max_grid_size).type(torch.float32).contiguous())    # (batch * block_size, h_dim)
        s_emb = s_emb.reshape(states.shape[0], states.shape[1], 1, self.h_dim)                          # (batch, block_size, n_embd)
        
        # clip 임베딩
        clip_emb = self.state_emb_layer(clip.reshape(-1, 1, self.max_grid_size, self.max_grid_size).type(torch.float32).contiguous())
        clip_emb = clip_emb.reshape(clip.shape[0], clip.shape[1], 1, self.h_dim)
        
        # in_grid 임베딩
        in_grid_emb = self.state_emb_layer(in_grid.reshape(-1, 1, self.max_grid_size, self.max_grid_size).type(torch.float32).contiguous())
        in_grid_emb = in_grid_emb.reshape(in_grid.shape[0], in_grid.shape[1], 1, self.h_dim)
        in_grid_tiled = in_grid_emb.tile([1, s_emb.shape[1], 1, 1])       # 차원 맞춰주려고 N번 반복
        
        # input-output pair 임베딩
        pair = torch.cat([pair_in, pair_out], dim=1)
        pair_emb = self.state_emb_layer(pair.reshape(-1, 1, self.max_grid_size, self.max_grid_size).type(torch.float32).contiguous())
        pair_emb = pair_emb.reshape(pair.shape[0], 1, 1, pair.shape[1]*self.h_dim)
        pair_emb = self.pair_emb_layer(pair_emb)
        pair_tiled = pair_emb.tile([1, s_emb.shape[1], 1, 1])       # 차원 맞춰주려고 N번 반복
        
        # action 임베딩
        a_emb = self.action_emb_layer(actions)
        x_emb = self.x_emb_layer(selection[:, :, 0])
        y_emb = self.y_emb_layer(selection[:, :, 1])
        h_emb = self.h_emb_layer(selection[:, :, 2])
        w_emb = self.w_emb_layer(selection[:, :, 3])

        # through rnn
        s_emb_a = torch.cat([s_emb, clip_emb, in_grid_tiled, pair_tiled, a_emb, x_emb, y_emb, h_emb, w_emb], dim=-2)
        s_emb_a_shape = s_emb_a.shape
        s_emb_a = s_emb_a.view(s_emb_a_shape[0], -1, s_emb_a_shape[-1]).contiguous()
        
        feats, _ = self.rnn(s_emb_a)
        hn = feats[:,-1:,:]
        z_mean = self.mean_layer(hn)
        z_sig = self.sig_layer(hn)

        if self.normalize_latent:
            z_mean = z_mean/torch.norm(z_mean, dim=-1).unsqueeze(-1)
        
        return z_mean, z_sig
        

class Decoder(nn.Module):
    '''
    Decoder module.
    Decoder takes states, actions, and a sampled z and outputs parameters of P(s_T|s_0,z) and P(a_t|s_t,z) for all t in {0,...,T}
    P(s_T|s_0,z) is our "abstract dynamics model", because it predicts the resulting state transition over T timesteps given a skill 
    (so similar to regular dynamics model, but in skill space and also temporally extended)
    P(a_t|s_t,z) is our "low-level policy", so this is the feedback policy the agent runs while executing skill described by z.
    We can try the following architecture:
    -embed z
    -Pass into fully connected network to get "state T features"
    '''
    def __init__(self,state_dim,a_dim,z_dim,h_dim,a_dist,fixed_sig,state_decoder_type,policy_decoder_type,per_element_sigma,
                 action_num,max_grid_size,state_emb_layer,pair_emb_layer):

        super(Decoder,self).__init__()
        
        self.state_emb_layer = state_emb_layer
        self.pair_emb_layer = pair_emb_layer
        
        print('in decoder a_dist: ', a_dist)
        self.state_dim = state_dim
        self.a_dim = a_dim
        self.z_dim = z_dim

        if state_decoder_type == 'mlp':
            self.abstract_dynamics = AbstractDynamics(state_dim,z_dim,h_dim,per_element_sigma=per_element_sigma)
        elif state_decoder_type == 'autoregressive':
            self.abstract_dynamics = AutoregressiveStateDecoder(state_dim,z_dim,h_dim)

        if policy_decoder_type == 'mlp':
            self.ll_policy = LowLevelPolicy(state_dim, a_dim, z_dim, h_dim, a_dist, action_num, max_grid_size, fixed_sig=fixed_sig, 
                                            state_emb_layer=self.state_emb_layer, pair_emb_layer=self.pair_emb_layer)
        elif policy_decoder_type == 'autoregressive':
            self.ll_policy = AutoregressiveLowLevelPolicy(state_dim,a_dim,z_dim,h_dim,a_dist=a_dist,fixed_sig=None)

        # self.emb_layer  = nn.Linear(state_dim+z_dim,h_dim)
        # self.fc = nn.Sequential(nn.Linear(state_dim+z_dim,h_dim),nn.ReLU(),nn.Linear(h_dim,h_dim),nn.ReLU())

        self.state_decoder_type = state_decoder_type
        self.policy_decoder_type = policy_decoder_type
        self.a_dist = a_dist

        
    def forward(self, states, clip, in_grid, actions, selection, z, pair_in, pair_out, state_decoder):

        '''
        INPUTS: 
            states: batch_size x T x state_dim state sequence tensor
            z:      batch_size x 1 x z_dim sampled z/skill variable
        OUTPUTS:
            sT_mean: batch_size x 1 x state_dim tensor of terminal (time=T) state means
            sT_sig:  batch_size x 1 x state_dim tensor of terminal (time=T) state standard devs
            a_mean: batch_size x T x a_dim tensor of action means for each t in {0.,,,.T}
            a_sig:  batch_size x T x a_dim tensor of action standard devs for each t in {0.,,,.T}
        '''
        # state decoder 쓰는 경우
        s_0 = states[:,0:1,:]

        # MLP로 구현
        a_mean, a_sig, x_mean, x_sig, y_mean, y_sig, h_mean, h_sig, w_mean, w_sig = self.ll_policy(states, clip, in_grid, z, pair_in, pair_out)
        
        if state_decoder:
            s_T = states[:,-1:,:]
            if self.state_decoder_type == 'autoregressive':
                sT_mean, sT_sig = self.abstract_dynamics(s_0, s_T, z.detach())
            elif self.state_decoder_type == 'mlp':
                sT_mean, sT_sig = self.abstract_dynamics(s_0, z.detach())
            return sT_mean, sT_sig, a_mean, a_sig, x_mean, x_sig, y_mean, y_sig, h_mean, h_sig, w_mean, w_sig

        else:
            return a_mean, a_sig, x_mean, x_sig, y_mean, y_sig, h_mean, h_sig, w_mean, w_sig


class Prior(nn.Module):
    '''
    Decoder module.
    Decoder takes states, actions, and a sampled z and outputs parameters of P(s_T|s_0,z) and P(a_t|s_t,z) for all t in {0,...,T}
    P(s_T|s_0,z) is our "abstract dynamics model", because it predicts the resulting state transition over T timesteps given a skill 
    (so similar to regular dynamics model, but in skill space and also temporally extended)
    P(a_t|s_t,z) is our "low-level policy", so this is the feedback policy the agent runs while executing skill described by z.
    We can try the following architecture:
    -embed z
    -Pass into fully connected network to get "state T features"
    '''
    def __init__(self,state_dim,z_dim,h_dim,goal_conditioned=False,goal_dim=2,max_grid_size=30,state_emb_layer=None,pair_emb_layer=None):

        super(Prior,self).__init__()
        
        self.state_dim = state_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.goal_conditioned = goal_conditioned
        if(self.goal_conditioned):
            self.goal_dim = goal_dim
        else:
            self.goal_dim = 0
        self.layers = nn.Sequential(
            nn.Linear(state_dim+state_dim+state_dim+h_dim+self.goal_dim, h_dim),  # state, clip, pair
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU()
        )
        #self.mean_layer = nn.Linear(h_dim,z_dim)
        self.mean_layer = nn.Sequential(nn.Linear(h_dim,h_dim),nn.ReLU(),nn.Linear(h_dim,z_dim))
        #self.sig_layer  = nn.Sequential(nn.Linear(h_dim,z_dim),nn.Softplus())
        self.sig_layer  = nn.Sequential(nn.Linear(h_dim,h_dim),nn.ReLU(),nn.Linear(h_dim,z_dim),nn.Softplus())

        self.state_emb_layer = state_emb_layer
        self.pair_emb_layer = pair_emb_layer
        self.max_grid_size = max_grid_size
    def forward(self, s0, clip0, in_grid, pair_in, pair_out, goal=None):

        '''
        INPUTS: 
            states: batch_size x T x state_dim state sequence tensor
            
        OUTPUTS:
            z_mean: batch_size x 1 x state_dim tensor of z means
            z_sig:  batch_size x 1 x state_dim tensor of z standard devs
            
        '''
        # state 임베딩
        s_emb = self.state_emb_layer(s0.reshape(-1, 1, self.max_grid_size, self.max_grid_size).type(torch.float32).contiguous())    # (batch * block_size, h_dim)
        s_emb = s_emb.reshape(s0.shape[0], s0.shape[1], self.h_dim) # (batch, block_size, n_embd)
        if(self.goal_conditioned):
            s_emb = torch.cat([s_emb, goal],dim=-1)
            
        # clip 임베딩
        clip_emb = self.state_emb_layer(clip0.reshape(-1, 1, self.max_grid_size, self.max_grid_size).type(torch.float32).contiguous())
        clip_emb = clip_emb.reshape(clip0.shape[0], clip0.shape[1], self.h_dim)
        
        # in_grid 임베딩
        in_grid_emb = self.state_emb_layer(in_grid.reshape(-1, 1, self.max_grid_size, self.max_grid_size).type(torch.float32).contiguous())
        in_grid_emb = in_grid_emb.reshape(in_grid.shape[0], in_grid.shape[1], self.h_dim)

        # input-output pair 임베딩        
        pair = torch.cat([pair_in, pair_out], dim=1)
        pair_shape = pair.shape
        # print("Prior - pair shape : {0}".format(pair_shape))
        pair_emb = self.state_emb_layer(pair.reshape(-1, 1, self.max_grid_size, self.max_grid_size).type(torch.float32).contiguous())
        # print("Prior - Before pair_emb : {0}".format(pair_emb.shape))
        pair_emb = pair_emb.reshape(pair_shape[0], 1, pair_shape[1]*self.h_dim)
        # print("Prior - After pair_emb : {0}".format(pair_emb.shape))
        pair_emb = self.pair_emb_layer(pair_emb)
        pair_tiled = pair_emb.tile([1, s_emb.shape[1], 1])       # 차원 맞춰주려고 N번 반복
        
        s_emb = torch.cat([s_emb, clip_emb, in_grid_emb, pair_tiled], dim=-1)    
            
        feats = self.layers(s_emb)
        # get mean and stand dev of action distribution
        z_mean = self.mean_layer(feats)
        z_sig  = self.sig_layer(feats)

        return z_mean, z_sig

    def get_loss(self, states, clip, in_grid, actions, pair_in, pair_out, goal=None):
        '''
        To be used only for low level action Prior training
        '''
        a_mean, a_sig = self.forward(states, clip, in_grid, pair_in, pair_out, goal)

        a_dist = Normal.Normal(a_mean, a_sig)
        return - torch.mean(a_dist.log_prob(actions))

class GenerativeModel(nn.Module):

    def __init__(self, decoder, prior):
        super().__init__()
        self.decoder = decoder
        self.prior = prior

    def forward(self):
        pass


class SkillModel(nn.Module):
    def __init__(self,state_dim,a_dim,z_dim,h_dim,horizon,a_dist='normal',beta=1.0,fixed_sig=None,encoder_type='gru',state_decoder_type='mlp',policy_decoder_type='mlp',
                 per_element_sigma=True,conditional_prior=True,train_diffusion_prior=False,normalize_latent=False,
                 color_num=11,action_num=36,max_grid_size=30,diffusion_steps=100, use_in_out=False ,diffusion_scale=1.0):
        super(SkillModel, self).__init__()

        self.state_dim = state_dim # state dimension
        self.a_dim = a_dim # action dimension
        self.z_dim = z_dim
        self.encoder_type = encoder_type
        self.state_decoder_type = state_decoder_type
        self.policy_decoder_type = policy_decoder_type
        self.conditional_prior = conditional_prior
        self.train_diffusion_prior = train_diffusion_prior
        self.diffusion_prior = None
        self.a_dist = a_dist
        self.normalize_latent = normalize_latent
        self.max_grid_size = max_grid_size
        self.diffusion_scale = diffusion_scale
        self.use_in_out = use_in_out
        
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

        if encoder_type == 'gru':
            self.encoder = GRUEncoder(state_dim,a_dim,z_dim,h_dim,normalize_latent=normalize_latent,
                                      color_num=color_num,action_num=action_num,max_grid_size=max_grid_size,
                                      state_emb_layer=self.state_emb_layer,
                                      pair_emb_layer=self.pair_emb_layer
                                      )
        elif encoder_type == 'transformer':
            self.encoder = TransformEncoder(state_dim,a_dim,z_dim,h_dim,horizon)

        self.decoder = Decoder(state_dim,a_dim,z_dim,h_dim, a_dist, fixed_sig=fixed_sig,state_decoder_type=state_decoder_type,
                               policy_decoder_type=policy_decoder_type,per_element_sigma=per_element_sigma, 
                               action_num=action_num, max_grid_size=max_grid_size, state_emb_layer=self.state_emb_layer, pair_emb_layer=self.pair_emb_layer)
        
        if conditional_prior:
            self.prior = Prior(state_dim, z_dim, h_dim, state_emb_layer=self.state_emb_layer, pair_emb_layer=self.pair_emb_layer,max_grid_size=max_grid_size)
            self.gen_model = GenerativeModel(self.decoder, self.prior)
            
        if self.train_diffusion_prior:
            nn_model = Model_mlp(
                x_shape = state_dim,
                n_hidden = 512, 
                y_dim = z_dim, 
                embed_dim = 128, 
                net_type ='unet',
                max_grid_size=self.max_grid_size,
                use_in_out=self.use_in_out,
                # net_type ='transformer'
            ).to('cuda')
            self.diffusion_prior = Model_Cond_Diffusion(
                nn_model,
                betas=(1e-4, 0.02),
                n_T=diffusion_steps,    # 원래 100
                device='cuda',
                x_dim=state_dim,
                y_dim=z_dim,
                drop_prob=0.0,
                guide_w=0.0,
                use_in_out=self.use_in_out,
            )

        self.beta = beta

    def forward(self, states, clip, in_grid, actions, selection, pair_in, pair_out, state_decoder):
        
        '''
        Takes states and actions, returns the distributions necessary for computing the objective function
        INPUTS:
            states: batch_size x T x state_dim state sequence tensor
            actions: batch_size x T x a_dim action sequence tensor
        OUTPUTS:
            s_T_mean:     batch_size x 1 x state_dim tensor of means of "decoder" distribution over terminal states
            S_T_sig:      batch_size x 1 x state_dim tensor of standard devs of "decoder" distribution over terminal states
            a_means:      batch_size x T x a_dim tensor of means of "decoder" distribution over actions
            a_sigs:       batch_size x T x a_dim tensor of stand devs
            z_post_means: batch_size x 1 x z_dim tensor of means of z posterior distribution
            z_post_sigs:  batch_size x 1 x z_dim tensor of stand devs of z posterior distribution 
        '''

        # STEP 1. Encode states and actions to get posterior over z
        z_post_means, z_post_sigs = self.encoder(states, clip, in_grid, actions, selection, pair_in, pair_out)        
        
        # STEP 2. sample z from posterior
        if not self.normalize_latent: 
            z_sampled = self.reparameterize(z_post_means, z_post_sigs)
        else:
            z_sampled = z_post_means

        # STEP 3. Pass z_sampled and states through decoder 
        if state_decoder:
            s_T_mean, s_T_sig, a_mean, a_sig, x_mean, x_sig, y_mean, y_sig, h_mean, h_sig, w_mean, w_sig = self.decoder(states, clip, in_grid, actions, selection, z_sampled, pair_in, pair_out, state_decoder)
            return s_T_mean, s_T_sig, a_mean, a_sig, x_mean, x_sig, y_mean, y_sig, h_mean, h_sig, w_mean, w_sig, z_post_means, z_post_sigs, z_sampled
        else:
            a_mean, a_sig, x_mean, x_sig, y_mean, y_sig, h_mean, h_sig, w_mean, w_sig = self.decoder(states, clip, in_grid, actions, selection, z_sampled, pair_in, pair_out, state_decoder)
            return a_mean, a_sig, x_mean, x_sig, y_mean, y_sig, h_mean, h_sig, w_mean, w_sig, z_post_means, z_post_sigs, z_sampled
    
    def get_losses(self, states, clip, in_grid, actions, selection, pair_in, pair_out, state_decoder):
        '''
        Computes various components of the loss:
        L = E_q [log P(s_T|s_0,z)] 
          + E_q [sum_t=0^T P(a_t|s_t,z)] 
          - D_kl(q(z|s_0,...,s_T,a_0,...,a_T)||P(z_0|s_0))
        Distributions we need:
        '''
        T = states.shape[1]
        # loss terms corresponding to -logP(s_T|s_0,z) and -logP(a_t|s_t,z)
        
        if state_decoder:   # 못씀
            s_T = states[:,-1:,:]
            s_T_mean, s_T_sig, a_mean, a_sig, x_mean, x_sig, y_mean, y_sig, h_mean, h_sig, w_mean, w_sig, z_post_means, z_post_sigs, z_sampled = self.forward(states, clip, in_grid, actions, selection, pair_in, pair_out, state_decoder)
            s_T_dist = Normal.Normal(s_T_mean, s_T_sig)
            s_T_loss = -torch.mean(torch.sum(s_T_dist.log_prob(s_T), dim=-1)) / T
        else:
            a_mean, a_sig, x_mean, x_sig, y_mean, y_sig, h_mean, h_sig, w_mean, w_sig, z_post_means, z_post_sigs, z_sampled = self.forward(states, clip, in_grid, actions, selection, pair_in, pair_out, state_decoder)
        
        # 원래 있던 Loss 쓰는 경우 -> 카테고리화
        a_dist = torch.distributions.categorical.Categorical(a_mean)
        x_dist = torch.distributions.categorical.Categorical(x_mean)
        y_dist = torch.distributions.categorical.Categorical(y_mean)
        h_dist = torch.distributions.categorical.Categorical(h_mean)
        w_dist = torch.distributions.categorical.Categorical(w_mean)
        
        z_post_dist = Normal.Normal(z_post_means, z_post_sigs)
        
        if not self.normalize_latent:
            if self.conditional_prior:
                z_prior_means, z_prior_sigs = self.prior(states[:,0:1,:], clip[:,0:1,:], in_grid, pair_in, pair_out) 
                z_prior_dist = Normal.Normal(z_prior_means, z_prior_sigs) 
            else:
                z_prior_means = torch.zeros_like(z_post_means)
                z_prior_sigs = torch.ones_like(z_post_sigs)
                z_prior_dist = Normal.Normal(z_prior_means, z_prior_sigs)
        
        # 원래 있던 Loss 쓰는 경우
        # print(actions.shape, actions.squeeze(-1).shape, selection.shape, selection[:, :, 0].squeeze(-1).shape)
        a_loss = -torch.mean(torch.sum(a_dist.log_prob(actions.squeeze(-1)), dim=-1))
        x_loss = -torch.mean(torch.sum(x_dist.log_prob(selection[:, :, 0].squeeze(-1)), dim=-1))
        y_loss = -torch.mean(torch.sum(y_dist.log_prob(selection[:, :, 1].squeeze(-1)), dim=-1))
        h_loss = -torch.mean(torch.sum(h_dist.log_prob(selection[:, :, 2].squeeze(-1)), dim=-1))
        w_loss = -torch.mean(torch.sum(w_dist.log_prob(selection[:, :, 3].squeeze(-1)), dim=-1))
        
        # KL loss 계산
        if not self.normalize_latent:
            kl_loss = torch.mean(torch.sum(KL.kl_divergence(z_post_dist, z_prior_dist), dim=-1))/T 
        else:
            kl_loss = torch.tensor(0.0).cuda()

        # Diffusion prior loss 계산
        if self.train_diffusion_prior:
            # diffusion_loss = self.diffusion_prior.loss_on_batch(states[:, 0:1, :, :], clip[:, 0:1, :, :], in_grid, z_sampled[:, 0, :].detach(), predict_noise=0)
            diffusion_loss = self.diffusion_prior.loss_on_batch(states[:, 0:1, :, :], clip[:, 0:1, :, :], in_grid, pair_in, pair_out, z_sampled[:, 0, :].detach(), predict_noise=0)
        else:
            diffusion_loss = 0.0
            
        loss_tot = (a_loss + x_loss + y_loss + h_loss + w_loss) + self.beta * kl_loss + diffusion_loss 

        if state_decoder:
            loss_tot += s_T_loss
            return  loss_tot, s_T_loss, a_loss, x_loss, y_loss,  h_loss, w_loss, kl_loss, diffusion_loss, 
        else:
            return  loss_tot, a_loss, x_loss, y_loss,  h_loss,  w_loss, kl_loss, diffusion_loss
    
    def reparameterize(self, mean, std):
        eps = torch.normal(torch.zeros(mean.size()).cuda(), torch.ones(mean.size()).cuda())
        return mean + std*eps


# """미사용 코드"""
class SkillPolicy(nn.Module):
    def __init__(self, state_dim, z_dim, h_dim):
        super(SkillPolicy,self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, z_dim)
        )

    def forward(self,state):

        return self.layers(state)