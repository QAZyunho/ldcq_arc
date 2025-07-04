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
CUDA_VISIBLE_DEVICES=1 python ./plan_skills_diffusion_ARCLE.py \
--env ARCLE/O2ARCv2Env-v0 \
--test_solar_dir /home/jovyan/beomi/jaehyun/ldcq_arc/ARC_Single/whole/test.10.09.13 \
--checkpoint_dir /home/jovyan/beomi/jaehyun/ldcq_arc/LDCQ_for_SOLAR/checkpoints/gpu2_09.15 \
--q_checkpoint_dir /home/jovyan/beomi/jaehyun/ldcq_arc/LDCQ_for_SOLAR/q_checkpoints/gpu2_09.13_0.35 \
--skill_model_filename Openhpc_gpu2_ARCLE_09.13_400_.pth \
--diffusion_model_filename Openhpc_gpu2_ARCLE_09.13_400__diffusion_prior_best.pt \
--policy_decoder_type mlp \
--num_diffusion_samples 100 \
--q_checkpoint_steps 150 \
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
--max_grid_size 10