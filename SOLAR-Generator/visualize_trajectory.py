import utils_visualize as utils
import json
import argparse
import os
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default="whole")  # segment, inout, gif
parser.add_argument('--file_path', nargs='+', type=str, required=True)  # JSON file path. Can be multiple arguments. If directory, fetch all child JSON files in that folder.
parser.add_argument('--save_folder_path', type=str, default="figure")  # folder to save figure
parser.add_argument('--make_task_folder', type=str, default="False")  # If True, make subfolder named with task id. If 'gif' mode, automatically make task id folder


args = parser.parse_args()


mode = args.mode.lower()
# data_folder_path = args.data_folder_path
file_path = args.file_path
save_folder_path = args.save_folder_path
make_task_folder = args.make_task_folder.lower()

file_path_list, json_list = utils.find_data(file_path)

if not file_path_list:
    raise ValueError('not correct path')

print("targets: ", json_list)

def parse_filename(file_name, mode):
    """
    Parse filename to extract task_id and trace_id based on different formats:
    - 74dd1130-gold_standard_1.json
    - 74dd1130-expert_1.json
    - 74dd1130-random_1.json
    - 74dd1130-gold_standard_1_0.json (for segment mode)
    - 74dd1130-expert_1_0.json (for segment mode) 
    - 74dd1130-random_1_0.json (for segment mode)
    """
    # Remove .json extension
    name_without_ext = file_name.replace('.json', '')
    
    if mode == "segment":
        # Handle segment format: 74dd1130-gold_standard_1_0 or 74dd1130-expert_1_0 or 74dd1130-random_1_0
        if 'gold_standard' in name_without_ext:
            parts = name_without_ext.split('-')
            if len(parts) >= 2:
                task_id = parts[0]
                remaining = '-'.join(parts[1:])  # gold_standard_1_0
                if remaining.startswith('gold_standard_'):
                    # Split by underscore: ['gold_standard', '1', '0']
                    remaining_parts = remaining.split('_')
                    if len(remaining_parts) >= 3:
                        trace_id = remaining_parts[1]  # '1'
                    else:
                        return None, None
                else:
                    return None, None
            else:
                return None, None
                
        elif 'expert' in name_without_ext:
            parts = name_without_ext.split('-')
            if len(parts) >= 2:
                task_id = parts[0]
                remaining = '-'.join(parts[1:])  # expert_1_0
                if remaining.startswith('expert_'):
                    # Split by underscore: ['expert', '1', '0']
                    remaining_parts = remaining.split('_')
                    if len(remaining_parts) >= 3:
                        trace_id = remaining_parts[1]  # '1'
                    else:
                        return None, None
                else:
                    return None, None
            else:
                return None, None
                
        elif 'random' in name_without_ext:
            parts = name_without_ext.split('-')
            if len(parts) >= 2:
                task_id = parts[0]
                remaining = '-'.join(parts[1:])  # random_1_0
                if remaining.startswith('random_'):
                    # Split by underscore: ['random', '1', '0']
                    remaining_parts = remaining.split('_')
                    if len(remaining_parts) >= 3:
                        trace_id = remaining_parts[1]  # '1'
                    else:
                        return None, None
                else:
                    return None, None
            else:
                return None, None
        else:
            # Fallback to original segment logic
            part = name_without_ext.split('_')
            if len(part) != 3:
                return None, None
            task_id = part[0]
            trace_id = part[1]
        
    elif mode in ["wrong", "whole", "inout", "gif"]:
        # Handle gold_standard, expert, and random formats
        if 'gold_standard' in name_without_ext:
            # Format: 74dd1130-gold_standard_1
            parts = name_without_ext.split('-')
            if len(parts) >= 2:
                task_id = parts[0]
                # Extract trace_id from the part after gold_standard_
                remaining = '-'.join(parts[1:])  # gold_standard_1
                if remaining.startswith('gold_standard_'):
                    trace_id = remaining.replace('gold_standard_', '')
                else:
                    trace_id = remaining.split('_')[-1]
            else:
                return None, None
                
        elif 'expert' in name_without_ext:
            # Format: 74dd1130-expert_1
            parts = name_without_ext.split('-')
            if len(parts) >= 2:
                task_id = parts[0]
                # Extract trace_id from the part after expert_
                remaining = '-'.join(parts[1:])  # expert_1
                if remaining.startswith('expert_'):
                    trace_id = remaining.replace('expert_', '')
                else:
                    trace_id = remaining.split('_')[-1]
            else:
                return None, None
                
        elif 'random' in name_without_ext:
            # Format: 74dd1130-random_1
            parts = name_without_ext.split('-')
            if len(parts) >= 2:
                task_id = parts[0]
                # Extract trace_id from the part after random_
                remaining = '-'.join(parts[1:])  # random_1
                if remaining.startswith('random_'):
                    trace_id = remaining.replace('random_', '')
                else:
                    trace_id = remaining.split('_')[-1]
            else:
                return None, None
        else:
            # Fallback to original logic for other formats
            part = name_without_ext.split('_')
            if len(part) != 2:
                return None, None
            task_id = part[0]
            trace_id = part[1]
    else:
        return None, None
        
    return task_id, trace_id

for i in tqdm(range(len(file_path_list)), position=0):
    path = file_path_list[i]
    file_name = path.split('/')[-1]

    task_id, trace_id = parse_filename(file_name, mode)
    
    if task_id is None or trace_id is None:
        print(f"Skipping file with incorrect format: {file_name}")
        continue

    with open(f"{path}", 'r') as file:
        data = json.load(file)

    utils.plot_task(mode, data, task_id, trace_id, save_folder_path, make_task_folder)

    if mode == 'gif':
        png_folder_path = f"{save_folder_path}/{task_id}/gif/pngs_{task_id}_{trace_id}"
        output_filename = f"{save_folder_path}/{task_id}/gif/{task_id}_{trace_id}.gif"
        utils.make_gif(png_folder_path, output_filename)