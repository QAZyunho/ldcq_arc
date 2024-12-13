CUDA_VISIBLE_DEVICES=0 python ./train_diffusion.py \
--env ARCLE \
--data_dir /home/jovyan/ldcq_arc/LDCQ_for_SOLAR/data/train.6f8cd79b.10.11.26 \
--checkpoint_dir /home/jovyan/ldcq_arc/LDCQ_for_SOLAR/checkpoints/gpu0_11.26 \
--skill_model_filename gpu0_skill_model_ARCLE_11.26_400_.pth \
--n_epoch 400 \
--save_cycle 10 \
--diffusion_steps 500 \
--gpu_name gpu0 \
--s_dim 512 \
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