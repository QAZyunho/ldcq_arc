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

for i in tqdm(range(len(file_path_list)), position=0):
    path = file_path_list[i]
    file_name = path.split('/')[-1]

    # print(path)
    # print(file_name)

    part = file_name.split('_')
    if mode == "segment":
        if len(part) != 3:
            print("not correct mode!")
            continue
        task_id = part[0]
        trace_id = part[1]

    elif mode == "wrong" or mode == "whole" or mode == "inout" or mode == "gif":
        if len(part) != 2:
            print("not correct mode!")
            continue
        task_id = file_name.split('_')[0]
        trace_id = file_name.split('.')[0].split('_')[1]
    else:
        raise NotImplementedError

    with open(f"{path}", 'r') as file:
        data = json.load(file)

    utils.plot_task(mode, data,  task_id, trace_id, save_folder_path, make_task_folder)

    if mode == 'gif':
        png_folder_path = f"{save_folder_path}/{task_id}/gif/pngs_{task_id}_{trace_id}"
        output_filename = f"{save_folder_path}/{task_id}/gif/{task_id}_{trace_id}.gif"
        utils.make_gif(png_folder_path, output_filename)