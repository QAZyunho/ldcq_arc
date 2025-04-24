import json
import os
import numpy as np
import importlib.util
from datetime import datetime


action_names = [  # mapping action number and action name of ARCLE
    "Color0",  # 0
    "Color1",  # 1
    "Color2",  # 2
    "Color3",  # 3
    "Color4",  # 4
    "Color5",  # 5
    "Color6",  # 6
    "Color7",  # 7
    "Color8",  # 8
    "Color9",  # 9
    "FloodFill0",  # 10
    "FloodFill1",  # 11
    "FloodFill2",  # 12
    "FloodFill3",  # 13
    "FloodFill4",  # 14
    "FloodFill5",  # 15
    "FloodFill6",  # 16
    "FloodFill7",  # 17
    "FloodFill8",  # 18
    "FloodFill9",  # 19
    "MoveU",  # 20
    "MoveD",  # 21
    "MoveR",  # 22
    "MoveL",  # 23
    "Rotate90",  # 24
    "Rotate270",  # 25
    "FlipH",  # 26
    "FlipV",  # 27
    "CopyI",  # 28
    "CopyO",  # 29
    "Paste",  # 30
    "CopyInput",  # 31
    "ResetGrid",  # 32
    "ResizeGrid",   # 33, in 'ARCLE/O2ARCv2Env-v0', 'ReszieGrid' is labeled as 'CropGrid'.
    "Submit",  # 34
    "None"  # 35
]


def sel_bbox_to_mask(selection_bbox, max_grid_dim):
    '''
    make selection mask grid( 1 for selected 0 for otherwise )
    x: starting height(row) of selection
    y: starting width(column) of selection
    h: number of selected unit squares on the height axis except for itself. ex) 0 means just select itself on the height axis.
    w: number of selected unit squares on the width axis except for itself. ex) 0 means just select itself on the width axis.
    '''
    x, y, h, w = selection_bbox
    sel_mask = np.zeros(max_grid_dim, dtype=np.int8)
    sel_mask[x:x+h+1, y:y+w+1] = 1

    return sel_mask


def mapping_operation(n):  # return action name of given integer
    try:
        return action_names[n]
    except:
        raise ValueError("not defined action number")

now = datetime.now()
formatted_time = now.strftime("%y.%m.%d")

def save_wrong(data, task, t, max_grid_dim, data_folder_path):  # save wrong(broken) trajectory

    whole_folder_path = data_folder_path+'/wrong'
    if not os.path.exists(whole_folder_path):
        os.makedirs(whole_folder_path)

    task_folder_path = whole_folder_path+f"/{t}.{task}.s{max_grid_dim}.{formatted_time}"
    if not os.path.exists(task_folder_path):
        os.makedirs(task_folder_path)

    with open(f"{task_folder_path}/{data['desc']['id']}.json", 'w') as f:
        json.dump(data, f)
        f.close()


def save_whole(data, task, t, max_grid_dim, data_folder_path):  # save whole trajectory

    whole_folder_path = data_folder_path+'/whole'
    if not os.path.exists(whole_folder_path):
        os.makedirs(whole_folder_path)

    task_folder_path = whole_folder_path+f"/{t}.{task}.s{max_grid_dim}.{formatted_time}"
    if not os.path.exists(task_folder_path):
        os.makedirs(task_folder_path)

    with open(f"{task_folder_path}/{data['desc']['id']}.json", 'w') as f:
        json.dump(data, f)
        f.close()


def save_seg(data, task, H, t, max_grid_dim, data_folder_path):  # save segment trajectory

    seg_folder_path = data_folder_path+'/segment'
    if not os.path.exists(seg_folder_path):
        os.makedirs(seg_folder_path)

    task_folder_path = seg_folder_path+f"/{t}.{task}.s{max_grid_dim}.{formatted_time}"
    if not os.path.exists(task_folder_path):
        os.makedirs(task_folder_path)

    sub_folder_path = task_folder_path+f"/{data['desc']['id']}"
    if not os.path.exists(sub_folder_path):
        os.makedirs(sub_folder_path)

    # make segment data from whole data then save
    for l in range(len(data['grid'])-H+1):
        seg_data = {
            "desc": {"id": data['desc']['id']+f"_{l}"},
            "step": data['step'][l:l+H],
            "selection": data['selection'][l:l+H],
            "operation": data['operation'][l:l+H],
            "operation_name": data['operation_name'][l:l+H],
            "reward": data['reward'][l:l+H],
            "terminated": data['terminated'][l:l+H],
            "grid_dim": data['grid_dim'][l:l+H],
            "in_grid": data['in_grid'],
            "out_grid": data['out_grid'],
            "grid": data['grid'][l:l+H],
            "clip_dim": data['clip_dim'][l:l+H],
            "clip": data['clip'][l:l+H],
            "selection_mask": data['selection_mask'][l:l+H],
            "ex_in": data['ex_in'],
            "ex_out": data['ex_out'],
            "ex_in_grid_dim": data['ex_in_grid_dim'],
            "ex_out_grid_dim": data['ex_out_grid_dim'],
            "next_grid": data['grid'][l+H-1 if l+H == len(data['grid']) else l+H],
            "next_clip": data['clip'][l+H-1 if l+H == len(data['grid']) else l+H]
        }

        with open(f"{sub_folder_path}/{seg_data['desc']['id']}.json", 'w') as f:
            json.dump(seg_data, f)
            f.close()


def append_data(data, grid_pad, sel, g_h, g_w, clip_pad, c_h, c_w, sel_mask, operation, reward, term, step):
    data['grid'].append(grid_pad.tolist())
    data['grid_dim'].append([int(g_h), int(g_w)])
    data['clip'].append(clip_pad.tolist())
    data['clip_dim'].append([int(c_h), int(c_w)])
    data['selection'].append(sel)
    data['selection_mask'].append(sel_mask.tolist())
    data['operation'].append(operation)
    data['operation_name'].append(mapping_operation(operation))
    data['reward'].append(reward)
    data['terminated'].append(term)
    data['step'].append(step)


def append_example(data, ei, eo, grid):
    hi, wi = ei.shape
    ho, wo = eo.shape
    grid_ex_in = grid.copy()
    grid_ex_in[:hi, :wi] = ei
    data['ex_in'].append(grid_ex_in.tolist())
    data['ex_in_grid_dim'].append([hi, wi])
    grid_ex_out = grid.copy()
    grid_ex_out[:ho, :wo] = eo
    data['ex_out'].append(grid_ex_out.tolist())
    data['ex_out_grid_dim'].append([ho, wo])


def import_library_for_task(task, num_samples, max_grid_dim, num_examples, rand_seed):
    # import random grid maker from task folder
    spec = importlib.util.spec_from_file_location(
        'grid_maker', f"{os.path.dirname(os.path.abspath(__file__))}/maker/{task}/grid_maker.py")
    grid_maker_path = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(grid_maker_path)
    grid_maker = grid_maker_path.GridMaker(
        num_samples=num_samples, max_grid_dim=max_grid_dim, num_examples=num_examples, rand_seed=rand_seed)

    return grid_maker


def convert_npint_to_int(item):
    if isinstance(item, dict):
        # dictionaries: recursively calls a function for all key-value pairs
        return {k: convert_npint_to_int(v) for k, v in item.items()}
    elif isinstance(item, list):
        # list: recursively calls a function for all elements
        return [convert_npint_to_int(elem) for elem in item]
    elif isinstance(item, np.ndarray):
        return [convert_npint_to_int(elem) for elem in item]
    elif isinstance(item, np.integer):
        # convert np.int -> Python int
        return int(item)
    elif isinstance(item, np.floating):
        return int(item)
    else:
        # don't change in other types
        return item