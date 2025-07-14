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
CUDA_VISIBLE_DEVICES=4 python ./plan_skills_diffusion_ARCLE.py \
--env ARCLE/O2ARCv2Env-v0 \
--test_solar_dir /home/ubuntu/yunho/ldcq_arc/data/whole/test.74dd1130-mix.s10.25.06.30 \
--checkpoint_dir /home/ubuntu/yunho/ldcq_arc/LDCQ_for_SOLAR/checkpoints/gpu1_07.02 \
--skill_model_filename gpu1_skill_model_ARCLE_07.02_400_.pth \
--diffusion_model_filename gpu1_skill_model_ARCLE_07.02_400__diffusion_prior_best.pt \
--q_checkpoint_dir /home/ubuntu/yunho/ldcq_arc/LDCQ_for_SOLAR/q_checkpoints/gpu1_07.02_r0.9 \
--policy_decoder_type mlp \
--num_diffusion_samples 300 \
--q_checkpoint_steps 350 \
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
--horizon 1 \
--policy q \
--render None \
--beta 0.1 \
--max_grid_size 10 \
--use_in_out 0 \
