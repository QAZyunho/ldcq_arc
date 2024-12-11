#!/bin/bash

#change the arguments below
python visualize_trajectory.py \
    --mode whole \
    --file_path /home/jovyan/ldcq_arc/ARC_Single/whole/train.6f8cd79b.10.11.20/6f8cd79b-gold_standard_100.json /home/jovyan/ldcq_arc/ARC_Single/whole/train.6f8cd79b.10.11.20/6f8cd79b-gold_standard_50.json\
    --save_folder_path /home/jovyan/ldcq_arc/ARC_Single/figure \
    --make_task_folder true

:<<"OPTIONS"
explanation of arguments
-mode: visualization type. 'whole', 'segment', 'inout', 'wrong', 'gif'
-file_path: paths of JSON files to visualize
    If directory, fetch all child JSON files in that folder.
    JSON file name type:
    1) {task_id}_{trace_id}.json for 'whole', 'inout', 'wrong', 'gif' mode
    2) {task_id}_{trace_id}_{segment_id}.json for 'segment' mode
-save_folder_path: folder to save images.
--make_task_folder: 
    If True, make subfolder in save_folder_path named with task id. 
    Else, just save in save_folder_path
    If 'gif' mode, automatically make task id folder
OPTIONS
    
