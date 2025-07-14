echo "=================================="
echo "Script: $0"
echo "Started at: $(TZ='Asia/Seoul' date)"
echo "=================================="
echo ""
echo "Script contents:"
cat "$0"
echo ""
echo "=================================="
echo "Execution begins:"
echo "=================================="
CUDA_VISIBLE_DEVICES=7 python ./train_q_net.py \
--env ARCLE \
--solar_dir /home/ubuntu/yunho/ldcq_arc/data_expert/segment/train.4258a5f9-expert-colorfix.s10.H5.25.07.10 \
--data_dir /home/ubuntu/yunho/ldcq_arc/LDCQ_for_SOLAR/data/gpu7_07.11 \
--checkpoint_dir /home/ubuntu/yunho/ldcq_arc/LDCQ_for_SOLAR/checkpoints/gpu7_07.11 \
--skill_model_filename gpu7_skill_model_ARCLE_07.11_400_.pth \
--diffusion_model_filename gpu7_skill_model_ARCLE_07.11_400__diffusion_prior_best.pt \
--q_checkpoint_dir /home/ubuntu/yunho/ldcq_arc/LDCQ_for_SOLAR/q_checkpoints/gpu7_07.11 \
--total_prior_samples 100 \
--num_prior_samples 100 \
--n_epoch 800 \
--diffusion_steps 500 \
--gpu_name gpu7 \
--a_dim 36 \
--z_dim 256 \
--h_dim 512 \
--s_dim 512 \
--batch_size 32 \
--max_grid_size 10 \
--gamma 0.7 \
--horizon 5 \
--use_in_out 0


:<<"OPTIONS"
explanation of arguments
-env: RL environment. If you change this, the data type and functions are all changed. 
-solar_dir: train dataset directory.
-data_dir: diffusion model 학습하는 데 필요한 데이터를 저장하는 directory.
-checkpoint_dir: vae 모델 저장된 directory.
-q_checkpoint_dir: dqn 모델 저장될 directory.
-skill_model_filename: vae 모델 파일 이름. pth 확장자로 된 것.
-diffusion_model_filename: diffusion 모델 파일 이름. checkpoint에 같이 저장된 pt 확장자 파일 중 선택.
-total_prior_samples, num_prior_samples sample 몇개 뽑을 지인데 두 개 동일하게 사용하면 됨. default는 500.
-diffusion_steps: 3번 diffusion model 학습에 사용한 diffusion model.
-horizon: step length of segment trace
-a_dim: operation 갯수 ARCLE 기준 0~34, 35는 None
-z_dim: size of latent
-h_dim: hidden layer. usually 2*z_dim
-skill_model_diffusion_steps 1~2번에서 사용한 skill model 학습시 사용한 diffusion step.
-train_diffusion_prior: vae 학습 시 diffusion prior도 같이 학습 시킬것인지. true로 하는 것이 vae 학습에 도움이 된다고 함.
-conditional_prior: vae에서 prior모듈을 별도로 학습시킬 것인지. default는 true.
-normalize_latent: latent normalize할 것인지. default는 true.
-gamma: discount factor.
-max_grid_size: maximum grid dim h
OPTIONS