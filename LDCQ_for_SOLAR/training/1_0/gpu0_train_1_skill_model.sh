CUDA_VISIBLE_DEVICES=0 python ./train_skills.py \
--env ARCLE \
--solar_dir /home/jovyan/ldcq_arc/ARC_Single/segment/train.6f8cd79b.10.11.26  \
--test_solar_dir /home/jovyan/ldcq_arc/ARC_Single/segment/test.6f8cd79b.10.11.26  \
--checkpoint_dir /home/jovyan/ldcq_arc/LDCQ_for_SOLAR/checkpoints \
--num_epochs 401 \
--start_training_state_decoder_after 402 \
--state_decoder_type mlp \
--test_on True \
--horizon 5 \
--a_dim 36 \
--z_dim 256 \
--h_dim 512 \
--s_dim 512 \
--gpu_name gpu0 \
--train_diffusion_prior 1 \
--conditional_prior 1 \
--normalize_latent 0 \
--diffusion_steps 100 \
--beta 0.1 \
--test_num 100 \
--test_cycle 10 \
--save_cycle 20 \
--batch_size 128 \
--max_grid_size 10 \
--date 11.26

:<<"OPTIONS"
explanation of arguments
-env: RL environment. If you change this, the data type and functions are all changed. 
-solar_dir: train dataset directory.
-test_solar_dir: validation dataset directory.
-checkpoint_dir: vae 모델 저장되는 directory. gpu 번호와 date에 맞는 하위 폴더 생성됨.
-start_training_state_decoder_after: when starting state decoder. For SOLAR, model doesn't use state decoder.
-test_on: validate or not. decoder accuracy를 측정해서 어느 정도 encoding-decoding이 되야 이후 단계가 진행될 수 있기에 키는 것을 추천.
-horizon: step length of segment trace
-a_dim: operation 갯수 ARCLE 기준 0~34, 35는 None
-z_dim: size of latent
-h_dim: hidden layer. usually 2*z_dim
-s_dim: state embedding layer usaually same with h_dim.
-train_diffusion_prior: vae 학습 시 diffusion prior도 같이 학습 시킬것인지. true로 하는 것이 vae 학습에 도움이 된다고 함.
-conditional_prior: vae에서 prior모듈을 별도로 학습시킬 것인지. default는 true.
-normalize_latent: latent normalize할 것인지. default는 true.
-diffusion_steps: diffusion prior 학습 시 diffusion step
-beta: beta vae학습 시 loss term 조절하는 파라미터. 
-max_grid_size: maximum grid dim h  
OPTIONS