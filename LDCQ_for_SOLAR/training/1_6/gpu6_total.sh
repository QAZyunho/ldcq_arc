# #!/bin/bash

# # 첫 번째 스크립트 실행
# nohup ./1_6/gpu6_train_1_skill_model.sh > ./log/gpu6_07.14/gpu6_1.log 2>&1 &
# # PID 저장
# PID_1=$!
# # 첫 번째 스크립트가 종료될 때까지 기다림
# wait $PID_1
# echo "First script finished."

# # 두 번째 스크립트 실행
# nohup ./1_6/gpu6_train_2_collect_diffusion_data.sh > ./log/gpu6_07.14/gpu6_2.log 2>&1 &
# # PID 저장
# PID_2=$!
# # 두 번째 스크립트가 종료될 때까지 기다림
# wait $PID_2
# echo "Second script finished."

# # 세 번째 스크립트 실행
# nohup ./1_6/gpu6_train_3_diffusion.sh> ./log/gpu6_07.14/gpu6_3.log 2>&1 &
# # PID 저장
# PID_4=$!
# # 세 번째 스크립트가 종료될 때까지 기다림
# wait $PID_4
# echo "Third script finished."

# 네 번째 스크립트 실행
nohup ./1_6/gpu6_train_4_collect_q_learning.sh > ./log/gpu6_07.14/gpu6_4.log 2>&1 &
# PID 저장
PID_4=$!
# 네 번째 스크립트가 종료될 때까지 기다림
wait $PID_4
echo "Fourth script finished."

nohup ./1_6/gpu6_train_5_q_learning.sh > ./log/gpu6_07.14/gpu6_5.log 2>&1 &
# PID 저장
PID_6=$!
# 네 번째 스크립트가 종료될 때까지 기다림
wait $PID_6
echo "Fifth script finished."