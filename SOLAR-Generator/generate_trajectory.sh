#!/bin/bash

#change the arguments below
python generate_trajectory.py \
    --env ARCLE/O2ARCv2Env-v0 \
    --data_folder_path /home/jovyan/beomi/jaehyun/ldcq_arc/ARC_Single \
    --tasks 5c0a986e-mix-colordiff \
    --num_samples 1000 \
    --num_examples 3 \
    --max_grid_dim 10 10 \
    --horizon 5 \
    --save_whole_trace True \
    --save_seg_trace True \
    --render_mode None \
    --delete_existing_data True \
    --rand_seed 0


:<<"OPTIONS"
explanation of arguments
-env: RL environment. If you change this, the data type and functions are all changed. 
-tasks: 
    1) task_id_1 task_id_2 .... :list of task_ids.
    2) tasks.txt : A file that contains one task ID per line.
    3) all : all tasks in the  'maker' folder
-num_samples: number of samples to generate for each task
-num_examples: number of example pairs for each trace data
-max_grid_dim: maximum grid dim h, w  
-horizon: step length of segment trace
-save_whole_trace: whether save the whole trace or not
-save_seg_trace: whether save segment trace or not
-render_mode: 'none' for generating trace quickly, 'ansi' for watching the step of generating trace.
-delete_existing_data: whether delete existing trace or not
-data_folder_path: path to save trace data files.
-rand_seed: random seed that all grid_makers share.
OPTIONS