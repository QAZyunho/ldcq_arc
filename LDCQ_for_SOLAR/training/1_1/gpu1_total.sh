#!/bin/bash

# # 첫 번째 스크립트 실행
# nohup ./1_1/gpu1_train_1_skill_model.sh > ./log/gpu1_07.02/gpu1_1.log 2>&1 &
# # PID 저장
# PID_1=$!
# # 첫 번째 스크립트가 종료될 때까지 기다림
# wait $PID_1
# echo "First script finished."

# # 두 번째 스크립트 실행
# nohup ./1_1/gpu1_train_2_collect_diffusion_data.sh > ./log/gpu1_07.02/gpu1_2.log 2>&1 &
# # PID 저장
# PID_2=$!
# # 두 번째 스크립트가 종료될 때까지 기다림
# wait $PID_2
# echo "Second script finished."

# # 세 번째 스크립트 실행
# nohup ./1_1/gpu1_train_3_diffusion.sh> ./log/gpu1_07.02/gpu1_3.log 2>&1 &
# # PID 저장
# PID_3=$!
# # 세 번째 스크립트가 종료될 때까지 기다림
# wait $PID_3
# echo "Third script finished."

# 네 번째 스크립트 실행
nohup ./1_1/gpu1_train_4_collect_q_learning.sh > ./log/gpu1_07.02/gpu1_r0.9_4.log 2>&1 &
# PID 저장
PID_4=$!
# 네 번째 스크립트가 종료될 때까지 기다림
wait $PID_4
echo "Fourth script finished."

nohup ./1_1/gpu1_train_5_q_learning.sh > ./log/gpu1_07.02/gpu1_r0.9_5.log 2>&1 &
# PID 저장
PID_5=$!
# 네 번째 스크립트가 종료될 때까지 기다림
wait $PID_5
echo "Fifth script finished."