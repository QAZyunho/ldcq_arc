import os
import sys
import datetime

curr_folder=os.path.abspath(__file__)
parent_folder=os.path.dirname(os.path.dirname(curr_folder))
sys.path.append(parent_folder) 

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader
import torch.distributions.normal as Normal
from models.skill_model import SkillModel
# import h5py		안씀
from utils.utils import get_dataset, ARC_Segment_Dataset
import pickle
from tqdm import tqdm
import argparse
import wandb

def train(model, optimizer, train_loader, train_state_decoder):

	losses = []

	pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc="Train skill model : ")

	for _, (state, s_T, clip, clip_T, selection, operation, reward, terminated, _, in_grid, out_grid, ex_in, ex_out) in pbar:
    	
		states = state.cuda()
		clip = clip.cuda()
		in_grid = in_grid.cuda()
		actions = operation.cuda()
		selection = selection.cuda()
		pair_in = ex_in.cuda()
		pair_out = ex_out.cuda()

		if train_state_decoder:
			loss_tot, s_T_loss, a_loss, x_loss, y_loss,  h_loss, w_loss, kl_loss, diffusion_loss= model.get_losses(states, clip, in_grid, actions, selection, pair_in, pair_out, train_state_decoder)

		else:
			loss_tot, a_loss, x_loss, y_loss,  h_loss, w_loss, kl_loss, diffusion_loss = model.get_losses(states, clip, in_grid, actions, selection, pair_in, pair_out, train_state_decoder)

		model.zero_grad()
		loss_tot.backward()
		optimizer.step()
  
		# log losses
		wandb.log({"train_skill/loss": loss_tot.item()})
		wandb.log({"train_skill/a_loss": a_loss.item()})
		wandb.log({"train_skill/x_loss": x_loss.item()})
		wandb.log({"train_skill/y_loss": y_loss.item()})
		wandb.log({"train_skill/h_loss": h_loss.item()})
		wandb.log({"train_skill/w_loss": w_loss.item()})
		wandb.log({"train_skill/kl_loss": kl_loss.item()})
		wandb.log({"train_skill/diffusion_loss": diffusion_loss.item() if train_diffusion_prior else diffusion_loss})

		losses.append(loss_tot.item())

		# pbar.set_description('Loss: %.3f' % loss_tot.item())
	
	mean_losses = np.mean(losses)

	wandb.log({"train_skill/mean_loss": mean_losses})
 
	return mean_losses

def test(model, test_loader, test_state_decoder, test_num):
    
	losses = []
	s_T_losses = []
	a_losses = []
	kl_losses = []
	s_T_ents = []
	diffusion_losses = []

	with torch.no_grad():  
		pbar = tqdm(enumerate(test_loader), total=test_num, desc="Test skill model : ")

		for i, (state, s_T, clip, clip_T, selection, operation, reward, terminated, _, in_grid, out_grid, ex_in, ex_out) in pbar:
			if(i >= test_num):
				break
			states = state.cuda()
			clip = clip.cuda()
			in_grid = in_grid.cuda()
			actions = operation.cuda()
			selection = selection.cuda()
			pair_in = ex_in.cuda()
			pair_out = ex_out.cuda()
   
			if test_state_decoder:
				loss_tot, s_T_loss, a_loss, x_loss, y_loss,  h_loss, w_loss, kl_loss, diffusion_loss = model.get_losses(states, clip, in_grid, actions, selection, pair_in, pair_out, test_state_decoder)
				s_T_losses.append(s_T_loss.item())
			else:
				loss_tot, a_loss, x_loss, y_loss,  h_loss, w_loss, kl_loss, diffusion_loss = model.get_losses(states, clip, in_grid, actions, selection, pair_in, pair_out, test_state_decoder)
			# log losses
			losses.append(loss_tot.item())
			a_losses.append(a_loss.item())
			kl_losses.append(kl_loss.item())
			diffusion_losses.append(diffusion_loss.item() if train_diffusion_prior else diffusion_loss)

	mean_losses = np.mean(losses)

	wandb.log({"train_skill/test_loss": mean_losses})
 
	if train_diffusion_prior:
		return np.mean(losses), np.mean(s_T_losses), np.mean(a_losses), np.mean(kl_losses), np.mean(diffusion_losses)
	return np.mean(losses), np.mean(s_T_losses), np.mean(a_losses), np.mean(kl_losses), None

def test_acc(model, test_loader, test_num):

	s_T_ents = []

	total_num = 0
	correct = 0
	correct_0 = 0
	total_0 = 0
 
	with torch.no_grad():
		pbar = tqdm(enumerate(test_loader), total=test_num, desc="Test Accuracy skill model : ")

		for i, (state, s_T, clip, clip_T, selection, operation, reward, terminated, _, in_grid, out_grid, ex_in, ex_out) in pbar:
			if(i >= test_num):
				break

			states = state.cuda()
			clip = clip.cuda()
			in_grid = in_grid.cuda()
			actions = operation.cuda()
			selection = selection.cuda()
			pair_in = ex_in.cuda()
			pair_out = ex_out.cuda()

			z_post_means, z_post_sigs = model.encoder(states, clip, in_grid, actions, selection, pair_in, pair_out)

			if not model.normalize_latent: 
				z_sampled = model.reparameterize(z_post_means, z_post_sigs)
			else:
				z_sampled = z_post_means

			total_0 += 1
			for h in range(H):
				pred_operation, pred_x, pred_y, pred_h, pred_w = model.decoder.ll_policy.tensor_policy(states[:, h, :, :], clip[:, h, :, :], in_grid, z_sampled, pair_in, pair_out)
				p_operation = torch.argmax(pred_operation)
				p_x = torch.argmax(pred_x)
				p_y = torch.argmax(pred_y)
				p_h = torch.argmax(pred_h)
				p_w = torch.argmax(pred_w)

				if (operation[0, h, 0] == p_operation and
					selection[0, h, 0] == p_x and
					selection[0, h, 1] == p_y and
					selection[0, h, 2] == p_h and
					selection[0, h, 3] == p_w):
					correct = correct + 1
					if h == 0:
						correct_0 += 1
    
				total_num = total_num + 1

	wandb.log({"train_skill/test_acc_whole": 100.0*correct/total_num})
	wandb.log({"train_skill/test_acc_s0": 100.0*correct_0/total_0})
	return correct/total_num

def test_acc_prior(model, test_loader, test_num):
    
	total_num = 0
	correct = 0
	with torch.no_grad():
		pbar = tqdm(enumerate(test_loader), total=test_num, desc="Test Accuracy skill model : ")

		for i, (state, s_T, clip, clip_T, selection, operation, reward, terminated, _, in_grid, out_grid, ex_in, ex_out) in pbar:
			if(i >= test_num):
				break

			states = state.cuda()
			clip = clip.cuda()
			in_grid = in_grid.cuda()
			actions = operation.cuda()
			selection = selection.cuda()
			pair_in = ex_in.cuda()
			pair_out = ex_out.cuda()

			# print("state_0 : {0}".format(states.shape))
			# print("clip_0 : {0}".format(clip.shape))
			# print("pair_in : {0}".format(pair_in.shape))
			# print("pair_out : {0}".format(pair_out.shape))
   
			latent, latent_prior_std = model.prior(states[:, 0:1, :, :], clip[:, 0:1, :, :], in_grid, pair_in, pair_out)

			# if not model.normalize_latent: 
			# 	z_sampled = model.reparameterize(z_post_means, z_post_sigs)
			# else:
			# 	z_sampled = z_post_means
   
			pred_operation, pred_x, pred_y, pred_h, pred_w = model.decoder.ll_policy.tensor_policy(states[:, 0, :, :], clip[:, 0, :, :], in_grid, latent, pair_in, pair_out)
			p_operation = torch.argmax(pred_operation)
			p_x = torch.argmax(pred_x)
			p_y = torch.argmax(pred_y)
			p_h = torch.argmax(pred_h)
			p_w = torch.argmax(pred_w)
   
			# print("operation : {0}, pred_operation : {1}".format(operation[0, 0, 0].shape, p_operation.shape))
			# print("x : {0}, p_x : {1}, p_x = {2}".format(selection[0, 0, 0].shape, p_x.shape, p_x))
			# print("y : {0}, p_y : {1}, p_y = {2}".format(selection[0, 0, 1].shape, p_y.shape, p_y))
			# print("h : {0}, p_h : {1}, p_h = {2}".format(selection[0, 0, 2].shape, p_h.shape, p_h))
			# print("w : {0}, p_w : {1}, p_w = {2}".format(selection[0, 0, 3].shape, p_w.shape, p_w))
   
			if (operation[0, 0, 0] == p_operation and
				selection[0, 0, 0] == p_x and
				selection[0, 0, 1] == p_y and
				selection[0, 0, 2] == p_h and
				selection[0, 0, 3] == p_w):
				correct = correct + 1
    
			total_num = total_num + 1

	wandb.log({"train_skill/test_prior_acc": 100.0*correct/total_num})
 
	return correct/total_num


parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='ARCLE')
parser.add_argument('--beta', type=float, default=0.1)		# 원래 0.05

parser.add_argument('--lr', type=float, default=5e-5)
parser.add_argument('--policy_decoder_type', type=str, default='mlp')	# 원래는 'autoregressive'
parser.add_argument('--state_decoder_type', type=str, default='mlp')
parser.add_argument('--a_dist', type=str, default='normal')
parser.add_argument('--horizon', type=int, default=5)
parser.add_argument('--separate_test_trajectories', type=int, default=0)
parser.add_argument('--test_on', type=bool, default=False)
parser.add_argument('--test_cycle', type=int, default=20)
parser.add_argument('--save_cycle', type=int, default=50)
parser.add_argument('--test_num', type=int, default=500)
parser.add_argument('--get_rewards', type=int, default=1)
parser.add_argument('--num_epochs', type=int, default=50000)
parser.add_argument('--start_training_state_decoder_after', type=int, default=10000)
parser.add_argument('--normalize_latent', type=int, default=0)

parser.add_argument('--append_goals', type=int, default=0)

parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--solar_dir', type=str, default=None)
parser.add_argument('--test_solar_dir', type=str, default=None)
parser.add_argument('--checkpoint_dir', type=str, default=parent_folder+'/checkpoints/')

parser.add_argument('--date', type=str, default='00.00')
parser.add_argument('--conditional_prior', type=int, default=1)
parser.add_argument('--train_diffusion_prior', type=int, default=1)
parser.add_argument('--diffusion_steps', type=int, default=500)

parser.add_argument('--gpu_name', type=str, required=True)
parser.add_argument('--optimizer', type=str, default="AdamW")

parser.add_argument('--a_dim', type=int, default=36)
parser.add_argument('--z_dim', type=int, default=128)		# 원래는 16
parser.add_argument('--h_dim', type=int, default=256)
parser.add_argument('--s_dim', type=int, default=256)
parser.add_argument('--max_grid_size', type=int, default=256)

args = parser.parse_args()

batch_size = args.batch_size #default 128

h_dim = args.h_dim
z_dim = args.z_dim
lr = args.lr #5e-5
wd = 0.0
H = args.horizon
stride = 1
n_epochs = args.num_epochs
# test_split = args.test_split
a_dist = args.a_dist	#'normal' # 'tanh_normal' or 'normal'
encoder_type = 'gru' 	# 'transformer' #'state_sequence'
state_decoder_type = args.state_decoder_type
policy_decoder_type = args.policy_decoder_type
load_from_checkpoint = False
per_element_sigma = True
start_training_state_decoder_after = args.start_training_state_decoder_after
train_diffusion_prior = args.train_diffusion_prior	# False
test_on = args.test_on
normalize_latent = args.normalize_latent

beta = args.beta # 1.0 # 0.1, 0.01, 0.001
conditional_prior = args.conditional_prior # True

checkpoint_dir = args.checkpoint_dir
env_name = args.env

action_num = args.a_dim
date = args.date

state_dim = args.s_dim # h_dim이랑 같은 것 사용
a_dim = args.a_dim # ARCLE에서 35개의 action

if(args.env == 'ARCLE'):
		dataset = ARC_Segment_Dataset(
		data_path=args.solar_dir
	)
		test_dataset = ARC_Segment_Dataset(
		data_path=args.test_solar_dir
	)
else:
    ValueError("지금은 ARCLE만 가능")

# 모델 저장할 때 이름
file_info = env_name  + '_' + date
filename = args.gpu_name+'_' + 'skill_model_' + file_info

# 확인하고 싶은 디렉토리의 경로를 지정하세요
checkpoint_dir = checkpoint_dir+'/'+args.gpu_name+'_'+date

# 디렉토리가 존재하는지 확인
if not os.path.exists(checkpoint_dir):
    # 디렉토리가 없으면 새로 만듭니다
    os.makedirs(checkpoint_dir)
    print(f"디렉토리가 생성되었습니다: {checkpoint_dir}")
else:
    print(f"디렉토리가 이미 존재합니다: {checkpoint_dir}")

# Check model option
print("Normalize_latent : {0}".format(normalize_latent))
print("Diffusion prior : {0}".format(train_diffusion_prior))
print("Conditional_prior : {0}".format(conditional_prior))

model = SkillModel(
    state_dim,
    a_dim,
    z_dim,
    h_dim,
    horizon=H,
    a_dist=a_dist,
    beta=beta,
    fixed_sig=None,
    encoder_type=encoder_type,
    state_decoder_type=state_decoder_type,
    policy_decoder_type=policy_decoder_type,
    per_element_sigma=per_element_sigma,
    conditional_prior=conditional_prior,
    train_diffusion_prior=train_diffusion_prior,
    diffusion_steps=args.diffusion_steps, 
    normalize_latent=normalize_latent,
    color_num=11,				# 0~9 색깔, 10은 배경
    action_num=action_num,		# 35개의 action
    max_grid_size=args.max_grid_size,
).cuda()



if(args.optimizer == "Adam"):
	optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
elif(args.optimizer == "AdamW"):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
else:
    ValueError("올바른 Optimizer를 입력해주세요")

d=datetime.datetime.now()
# Wandb 기록
wandb.init(
    project = "LDCQ_single",
    name = 'LDCQ_'+args.gpu_name+'_'+'skill'+'_'+str(d.month)+'.'+str(d.day)+'_'+str(d.hour)+'.'+str(d.minute),
    config = {
		'lr':lr,
		'h_dim':h_dim,
		'z_dim':z_dim,
		'H':H,
		'a_dim':a_dim,
		'state_dim':state_dim,
		'l2_reg':wd,
		'beta':beta,
		'env_name':env_name,
		'a_dist':a_dist,
		'filename':filename,
		'encoder_type':encoder_type,
		'state_decoder_type':state_decoder_type,
		'policy_decoder_type':policy_decoder_type,
		'per_element_sigma':per_element_sigma,
		'conditional_prior': conditional_prior,
		'train_diffusion_prior': train_diffusion_prior,
		# 'test_split': test_split,
		'separate_test_trajectories': args.separate_test_trajectories,
		'get_rewards': args.get_rewards,
		'normalize_latent': args.normalize_latent,
		'append_goals': args.append_goals
    }
)

train_loader = DataLoader(
	dataset=dataset,
	batch_size=batch_size,
	num_workers=0,
	shuffle=True)

test_loader = DataLoader(
	dataset=test_dataset,
	batch_size=1,
	num_workers=0,
	shuffle=True
 )

min_test_loss = 10**10
min_test_s_T_loss = 10**10
min_test_a_loss = 10**10
for i in range(n_epochs):
	# if(test_on and i % 50 == 0):
	if(test_on and i % args.test_cycle == 0):
		# Test Loss
		test_loss, test_s_T_loss, test_a_loss, test_kl_loss, test_diffusion_loss = test(model, train_loader, test_state_decoder = i > start_training_state_decoder_after, test_num=args.test_num)
		# test_loss, test_s_T_loss, test_a_loss, test_kl_loss, test_diffusion_loss = 0.0, 0.0, 0.0, 0.0, 0.0
		
		# Test Accuracy
		accuracy = test_acc(model, test_loader, test_num=args.test_num)
		# accuracy = test_acc(model, test_loader, test_num=args.test_num)
		prior_accuracy = test_acc_prior(model, test_loader, test_num=args.test_num)
  
		print("--------TEST---------")
		
		print('test_loss: ', test_loss)
		print('test_s_T_loss: ', test_s_T_loss)
		print('test_a_loss: ', test_a_loss)
		print('test_kl_loss: ', test_kl_loss)
		print('test_Acc: ', accuracy*100.0,'%')
		if test_diffusion_loss is not None:
			print('test_diffusion_loss ', test_diffusion_loss)
		print(i)
		if test_diffusion_loss is not None:
			pass
		
		if test_loss < min_test_loss:
			min_test_loss = test_loss	
			checkpoint_path = os.path.join(checkpoint_dir, filename+'_best.pth')
			torch.save({'model_state_dict': model.state_dict(),
					'optimizer_state_dict': optimizer.state_dict()}, checkpoint_path)
		if test_s_T_loss < min_test_s_T_loss:
			min_test_s_T_loss = test_s_T_loss

			checkpoint_path = os.path.join(checkpoint_dir, filename+'_best_sT.pth')
			torch.save({'model_state_dict': model.state_dict(),
					'optimizer_state_dict': optimizer.state_dict()}, checkpoint_path)
		if test_a_loss < min_test_a_loss:
			min_test_a_loss = test_a_loss

			checkpoint_path = os.path.join(checkpoint_dir, filename+'_best_a.pth')
			torch.save({'model_state_dict': model.state_dict(),
					'optimizer_state_dict': optimizer.state_dict()}, checkpoint_path)

	loss = train(model, optimizer, train_loader, train_state_decoder = i > start_training_state_decoder_after)
	
	print("--------TRAIN---------")
	
	print('Loss: ', loss)
	print("Epoch: {0}/{1}".format(i, n_epochs))
	# experiment.log_metric("Train loss", loss, step=i)

	# if i % 50 == 0:
	if i % args.save_cycle == 0:
		checkpoint_path = os.path.join(checkpoint_dir, filename+'_'+str(i)+'_'+'.pth')
		torch.save({'model_state_dict': model.state_dict(),
				'optimizer_state_dict': optimizer.state_dict()}, checkpoint_path)