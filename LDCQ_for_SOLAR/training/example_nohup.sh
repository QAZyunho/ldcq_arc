<1단계 - train_skills>
nohup ./1_0/gpu0_train_1_skill_model.sh > ./log/gpu0_04.10/gpu0_1.log 2>&1 &
nohup ./1_1/gpu1_train_1_skill_model.sh > ./log/gpu1_04.10/gpu1_1.log 2>&1 &
nohup ./1_2/gpu2_train_1_skill_model.sh > ./log/gpu2_04.10/gpu2_1.log 2>&1 &
nohup ./1_3/gpu3_train_1_skill_model.sh > ./log/gpu3_04.10/gpu3_1.log 2>&1 &


<2단계 - collect_diffusion_data>
nohup ./1_0/gpu0_train_2_collect_diffusion_data.sh > ./log/gpu0_09.12/gpu0_2.log 2>&1 &
nohup ./1_1/gpu1_train_2_collect_diffusion_data.sh > ./log/gpu1_09.12/gpu1.30_2.log 2>&1 &
nohup ./1_2/gpu2_train_2_collect_diffusion_data.sh > ./log/gpu2_09.12/gpu2_2_500.log 2>&1 &
nohup ./1_3/gpu3_train_2_collect_diffusion_data.sh > ./log/gpu3_09.12/gpu3_2.log 2>&1 &

nohup ./1_1_copy/gpu1_train_2_collect_diffusion_data.sh > ./log/gpu2_09.12/gpu2_2_1.log 2>&1 &
nohup ./1_1_copy/gpu1_train_3_diffusion.sh  > ./log/gpu2_09.12/gpu2_3_1.log 2>&1 &
nohup ./1_1_copy/gpu1_train_4_collect_q_learning.sh > ./log/gpu2_09.12/gpu1_4_0.5.log 2>&1 &
nohup ./1_0temp/gpu0_train_4_collect_q_learning.sh > ./log/gpu2_09.13/gpu1_4_0.5_100.log 2>&1 &
nohup ./1_0temp/gpu0_train_5_q_learning.sh > ./log/gpu2_09.13/gpu1_5_0.5_100.log 2>&1 &

<3단계 - train_diffusion>
nohup ./1_0/gpu0_train_3_diffusion.sh > ./log/gpu0_09.12/gpu0_3.log 2>&1 &
nohup ./1_1/gpu1_train_3_diffusion.sh > ./log/gpu1_09.12/gpu1.30_3.log 2>&1 &
nohup ./1_2/gpu2_train_3_diffusion.sh > ./log/gpu2_09.12/gpu2_3_500.log 2>&1 &
nohup ./1_3/gpu3_train_3_diffusion.sh > ./log/gpu3_09.12/gpu3_3.log 2>&1 &

<4단계 - collect_offline_q_learning_dataset>
nohup ./1_0/gpu0_train_4_collect_q_learning.sh > ./log/gpu0_09.12/gpu0_4.log 2>&1 &
nohup ./1_1/gpu1_train_4_collect_q_learning.sh > ./log/gpu0_09.10/gpu1_4.log 2>&1 &
nohup ./1_2/gpu2_train_4_collect_q_learning.sh > ./log/gpu2_09.13/gpu2_4_0.5.log 2>&1 &
nohup ./1_3/gpu3_train_4_collect_q_learning.sh > ./log/gpu3_09.12/gpu3_4.log 2>&1 &

<5단계 - train_q_net>
nohup ./1_0/gpu0_train_5_q_learning.sh > ./log/gpu0_09.10/gpu0_5.log 2>&1 &
nohup ./1_1/gpu1_train_5_q_learning.sh > ./log/gpu1_09.06/gpu1_5_0.6.log 2>&1 &
nohup ./1_2/gpu2_train_5_q_learning.sh > ./log/gpu3_09.10/gpu2_5_0.99.log 2>&1 &
nohup ./1_3/gpu3_train_5_q_learning.sh > ./log/gpu3_09.12/gpu3_5.log 2>&1 &

<eval>
nohup ./gpu0_test_ARCLE.sh > ./log/gpu0_04.10/gpu0_z32_1.log 2>&1 &
nohup ./gpu1_test_ARCLE.sh > ./log/gpu1_04.12_z64/gpu1_64_1.log 2>&1 &
nohup ./gpu2_test_ARCLE.sh > ./log/gpu2_04.10_z128/gpu2_vae_1.log 2>&1 &
nohup ./gpu3_test_ARCLE.sh > ./log/gpu3_09.12/gpu3_diffusion.log 2>&1 &

nohup ./gpu4_test_ARCLE.sh > ./log/gpu4_04.09/r0.5_H5_ex1_1.log 2>&1 &
nohup ./gpu5_test_ARCLE.sh > ./log/gpu5_04.09/r0.9_H5_ex1_1.log 2>&1 &

nohup ./gpu0.sh > ./log/gpu2_09.12/gpu0_q50.log 2>&1 &

nohup ./1_0/gpu0_total.sh > ./log/simple/gpu0_total_0.7.log 2>&1 &
nohup ./1_1/gpu1_total.sh > ./log/gpu1_04.12/gpu1_total.log 2>&1 &
nohup ./1_2/gpu2_total.sh > ./log/gpu2_04.10/gpu2_total.log 2>&1 &
nohup ./1_3/gpu3_total.sh > ./log/gpu3_04.10/gpu3_total.log 2>&1 &

ps -ef | grep train_skills.py
ps -ef | grep train_diffusion.py
ps aux | grep python

nohup ./gpu2_test_ARCLE.sh > ./log/gpu2_04.11/expert_r0.5_H5_ex5_1.log 2>&1 &