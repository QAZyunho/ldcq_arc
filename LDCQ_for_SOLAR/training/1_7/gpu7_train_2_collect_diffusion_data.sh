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

CUDA_VISIBLE_DEVICES=7 python ./collect_diffusion_data.py \
--env ARCLE \
--solar_dir /home/ubuntu/yunho/ldcq_arc/data_expert/segment/train.0d3d703e-expert.s10.H5.25.07.14 \
--data_dir /home/ubuntu/yunho/ldcq_arc/LDCQ_for_SOLAR/data/gpu7_07.14 \
--checkpoint_dir /home/ubuntu/yunho/ldcq_arc/LDCQ_for_SOLAR/checkpoints/gpu7_07.14 \
--skill_model_filename gpu7_skill_model_ARCLE_07.14_400_.pth \
--policy_decoder_type mlp \
--horizon 5 \
--a_dim 36 \
--z_dim 256 \
--h_dim 512 \
--s_dim 512 \
--train_diffusion_prior 1 \
--conditional_prior 1 \
--normalize_latent 0 \
--diffusion_steps 100 \
--max_grid_size 10 \
--use_in_out 1


:<<"OPTIONS"
explanation of arguments
-env: RL environment. If you change this, the data type and functions are all changed. 
-solar_dir: train dataset directory.
-data_dir: diffusion model 학습하는 데 필요한 데이터를 저장하는 directory.
-checkpoint_dir: vae 모델 저장된 directory.
-skill_model_filename: vae 모델 파일 이름. pth 확장자로 된 것.
-policy_decoder_type: mlp만 가능.
-horizon: step length of segment trace
-a_dim: operation 갯수 ARCLE 기준 0~34, 35는 None
-z_dim: size of latent
-h_dim: hidden layer. usually 2*z_dim
-s_dim: state embedding layer usaually same with h_dim.
-train_diffusion_prior: vae 학습 시 diffusion prior도 같이 학습 시킬것인지. true로 하는 것이 vae 학습에 도움이 된다고 함.
-conditional_prior: vae에서 prior모듈을 별도로 학습시킬 것인지. default는 true.
-normalize_latent: latent normalize할 것인지. default는 true.
-diffusion_steps: diffusion prior 학습 시 diffusion step
-max_grid_size: maximum grid dim h
OPTIONS