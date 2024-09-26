python ./plan_skills_diffusion_ARCLE.py \
    --env ARCLE/O2ARCv2Env-v0 \
    --dataset_dir /mnt/c/Users/Jaehyun/Desktop/Workspace/ldcq/ARC_data/test \
    --skill_model_filename "skill_model_ARCLE_3.14_16.31_50_.pth" \
    --q_checkpoint_dir 여기에 경로 넣기 \
    --state_decoder_type none \
    --q_checkpoint_steps 400 \
    --diffusion_steps 100 \
    --render None