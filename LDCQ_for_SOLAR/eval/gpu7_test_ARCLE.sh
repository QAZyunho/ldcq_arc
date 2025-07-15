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
CUDA_VISIBLE_DEVICES=7 python ./plan_skills_diffusion_ARCLE.py \
--env ARCLE/O2ARCv2Env-v0 \
--test_solar_dir /home/ubuntu/yunho/ldcq_arc/data_expert/whole/test.4258a5f9-expert-colorfix.s10.25.07.10 \
--checkpoint_dir /home/ubuntu/yunho/ldcq_arc/LDCQ_for_SOLAR/checkpoints/gpu7_07.11 \
--skill_model_filename gpu7_skill_model_ARCLE_07.11_400_.pth \
--diffusion_model_filename gpu7_skill_model_ARCLE_07.11_400__diffusion_prior_best.pt \
--q_checkpoint_dir /home/ubuntu/yunho/ldcq_arc/LDCQ_for_SOLAR/q_checkpoints/gpu7_07.11 \
--policy_decoder_type mlp \
--num_diffusion_samples 100 \
--q_checkpoint_steps 250 \
--diffusion_steps 500 \
--num_parallel_envs 1 \
--skill_model_diffusion_steps 100 \
--a_dim 36 \
--z_dim 256 \
--h_dim 512 \
--s_dim 512 \
--train_diffusion_prior 1 \
--conditional_prior 1 \
--normalize_latent 0 \
--exec_horizon 1 \
--horizon 5 \
--policy q \
--render None \
--beta 0.1 \
--max_grid_size 10 \
--use_in_out 1 \

    
:<<"OPTIONS"
explanation of arguments
-env: RL environment. If you change this, the data type and functions are all changed. 
-test_solar_dir: train dataset directory.
-checkpoint_dir: vae 모델 저장된 directory.
-skill_model_filename: vae 모델 파일 이름. pth 확장자로 된 것.
-diffusion_model_filename: diffusion 모델 파일 이름. checkpoint에 같이 저장된 pt 확장자 파일 중 선택.
-num_diffusion_samples: sample 몇개 뽑을 지인데 diffusion 썻냐 아니냐 차이에 따라 들어감. default는 500.
-q_checkpoint_steps: dqn 모델 선택하기. 항상 마지막이 더 잘 되는 것이 아님. loss graph보면 td error가 다시 올라가기에 적절히 선택할 필요.
-diffusion_steps: diffusion model에 사용한 diffusion step.
-horizon: step length of segment trace
-a_dim: operation 갯수 ARCLE 기준 0~34, 35는 None
-z_dim: size of latent
-h_dim: hidden layer. usually 2*z_dim
-s_dim: state embedding layer usaually same with h_dim.
-train_diffusion_prior: vae 학습 시 diffusion prior도 같이 학습 시킬것인지. true로 하는 것이 vae 학습에 도움이 된다고 함.
-conditional_prior: vae에서 prior모듈을 별도로 학습시킬 것인지. default는 true.
-normalize_latent: latent normalize할 것인지. default는 true.
-exec_horizon: 하나의 latent로 얼마만큼 action을 수행할지. LDCQ의 단점 중 하나로, 항상 여러 step을 고정적으로 수행하기에 1로 설정해서 하나씩 수행하는 것도 하나의 방법.
-horizon: latent에 들어가는 horizon.
-policy: q(LDCQ) / diffusion_prior(diffusion model 샘플링) / prior(VAE prior 샘플링). 
-beta: vae 학습 시 사용된 beta값
-max_grid_size: maximum grid dim h
OPTIONS