CUDA_VISIBLE_DEVICES=1 python ./train_diffusion.py \
--env ARCLE \
--solar_dir /home/jovyan/beomi/jaehyun/ldcq_arc/ARC_Single/segment/train.10.09.13  \
--data_dir /home/jovyan/beomi/jaehyun/ldcq_arc/LDCQ_for_SOLAR/data/gpu1_04.12 \
--checkpoint_dir /home/jovyan/beomi/jaehyun/ldcq_arc/LDCQ_for_SOLAR/checkpoints/gpu1_04.12_0 \
--skill_model_filename gpu1_skill_model_ARCLE_04.12_400_.pth \
--n_epoch 400 \
--save_cycle 10 \
--diffusion_steps 500 \
--gpu_name gpu1 \
--s_dim 256 \
--batch_size 32 \
--max_grid_size 10

:<<"OPTIONS"
explanation of arguments
-env: RL environment. If you change this, the data type and functions are all changed. 
-data_dir: diffusion model 학습하는 데 필요한 데이터를 저장하는 directory.
-checkpoint_dir: vae 모델 저장된 directory.
-skill_model_filename: vae 모델 파일 이름. pth 확장자로 된 것.
-diffusion_steps: diffusion model 학습 시 diffusion step. 앞의 이전단계들의 diffusion step과는 다른 것.
-s_dim: state embedding layer usaually same with h_dim.
-max_grid_size: maximum grid dim h
OPTIONS