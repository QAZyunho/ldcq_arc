#!/bin/bash

# # 첫 번째 스크립트 실행
# nohup ./1_4/gpu4_train_1_skill_model.sh > ./log/gpu4_07.11/gpu4_1.log 2>&1 &
# # PID 저장
# PID_1=$!
# # 첫 번째 스크립트가 종료될 때까지 기다림
# wait $PID_1
# echo "First script finished."

# # 두 번째 스크립트 실행
# nohup ./1_4/gpu4_train_2_collect_diffusion_data.sh > ./log/gpu4_07.11/gpu4_2.log 2>&1 &
# # PID 저장
# PID_2=$!
# # 두 번째 스크립트가 종료될 때까지 기다림
# wait $PID_2
# echo "Second script finished."

# # 세 번째 스크립트 실행
# nohup ./1_4/gpu4_train_3_diffusion.sh> ./log/gpu4_07.11/gpu4_3.log 2>&1 &
# # PID 저장
# PID_4=$!
# # 세 번째 스크립트가 종료될 때까지 기다림
# wait $PID_4
# echo "Third script finished."

# 네 번째 스크립트 실행
nohup ./1_4/gpu4_train_4_collect_q_learning.sh > ./log/gpu4_07.11/gpu4_4.log 2>&1 &
# PID 저장
PID_4=$!
# 네 번째 스크립트가 종료될 때까지 기다림
wait $PID_4
echo "Fourth script finished."

nohup ./1_4/gpu4_train_5_q_learning.sh > ./log/gpu4_07.11/gpu4_5.log 2>&1 &
# PID 저장
PID_5=$!
# 네 번째 스크립트가 종료될 때까지 기다림
wait $PID_5
echo "Fifth script finished."