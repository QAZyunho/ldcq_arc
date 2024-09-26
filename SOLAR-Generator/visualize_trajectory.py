import numpy as np
import gymnasium as gym
import utils
import time
import os
from tqdm import tqdm
import argparse
import shutil

parser = argparse.ArgumentParser()

parser.add_argument('--env', type=str, default='ARCLE/O2ARCv2Env-v0')
parser.add_argument('--tasks', nargs='+', type=str, required=True)  # task id // "all" for all tasks in maker folder// .txt file that contains task ids at each line
parser.add_argument('--data_folder_path', type=str, default="SOLAR_data")  # directory to save data, if there's no 'SOLAR_data' folder, 'SOLAR_data' folder will be made.
parser.add_argument('--num_samples', type=int, default=10000)  # number of samples to make in each tasks.
parser.add_argument('--num_examples', type=int, default=3)  # number of example pairs
parser.add_argument('--max_grid_dim', nargs=2, type=int, default=(30, 30))  # max grid dim h,w
parser.add_argument('--horizon', type=int, default=5)  # length(the number of steps) of one segment
parser.add_argument('--save_whole_trace', type=str, default="True")  # save whole data
parser.add_argument('--save_seg_trace', type=str, default="True")  # save segment data
# ansi if you want to see the real-time result, else do not set this argument
parser.add_argument('--render_mode', type=str, default="None")  # "ansi" to see the process
parser.add_argument('--delete_existing_data', type=str, default="False")  # make new trace with deleting old data// False for not save new data
parser.add_argument('--rand_seed', type=int, default=0)  # random seed that all grid_makers share.
args = parser.parse_args()

env_name = args.env
data_folder_path = args.data_folder_path
num_samples = args.num_samples
max_grid_dim = args.max_grid_dim
H = args.horizon
save_whole_trace = args.save_whole_trace.lower()
save_seg_trace = args.save_seg_trace.lower()
num_examples = args.num_examples
delete_existing_data = args.delete_existing_data.lower()
rand_seed = args.rand_seed

# set path to save data correctly.
# parts = data_folder_path.split('/')
# if parts[-1] == "SOLAR_data":
#     pass
# elif os.path.exists(data_folder_path+f'/SOLAR_data'):
#     print("There's already 'SOLAR_data' folder. please change folder path including 'SOLAR_data' folder or other folder to make 'SOLAR_data' folder.")
# else:
#     os.makedirs(data_folder_path+f'/SOLAR_data')
#     data_folder_path = data_folder_path+f'/SOLAR_data'

if args.render_mode.lower() == "none":
    render_mode = None
else:
    render_mode = args.render_mode.lower()

tasks = args.tasks
for file in tasks:
    if file == "all":
        task_list = [task for task in os.listdir('maker') if os.path.isdir(f'maker/{task}') and task != "__pycache__"]
        tasks = task_list

    elif ".txt" in file:
        tasks.remove(file)
        tasks_list = []
        with open(f'maker/{file}', 'r') as f:
            for line in f:
                tasks_list.append(line.strip())
        tasks.extend(tasks_list)

print("target: ", tasks)

for i in tqdm(range(len(tasks)), desc="total", position=0):

    if save_seg_trace == "true" or save_whole_trace == "true":
        if delete_existing_data != "true":
            whole_folder_path = data_folder_path+'/whole'+f"/{tasks[i]}"
            if os.path.exists(whole_folder_path):
                print(f"dataset exists already for {tasks[i]}")
                print("change '--delete_existing_data option' to delete old dataset then save new dataset")
                continue

            seg_folder_path = data_folder_path+'/segment'+f"/{tasks[i]}"
            if os.path.exists(seg_folder_path):
                print(f"dataset exists already for {tasks[i]}")
                print("change '--delete_existing_data option' to delete old dataset then save new dataset")
                continue

        else:
            whole_folder_path = data_folder_path+'/whole'+f"/{tasks[i]}"
            if os.path.exists(whole_folder_path):
                shutil.rmtree(whole_folder_path)
            seg_folder_path = data_folder_path+'/segment'+f"/{tasks[i]}"
            if os.path.exists(seg_folder_path):
                shutil.rmtree(seg_folder_path)
            wrong_folder_path = data_folder_path+'/wrong'+f"/{tasks[i]}"
            if os.path.exists(wrong_folder_path):
                shutil.rmtree(wrong_folder_path)

    try:
        grid_maker = utils.import_library_for_task(tasks[i], num_samples, max_grid_dim, num_examples=num_examples, rand_seed=rand_seed)
    except Exception as e:
        # falied to generate random grids, check if the max_grid_dim too small.
        print(e)
        print(f"For {tasks[i]}, failed to generate grid")
        continue

    # make environment
    env = gym.make(env_name, render_mode=render_mode, data_loader=grid_maker, max_grid_size=max_grid_dim, colors=10, max_episode_steps=None, max_trial=3)

    # iteration for all generated grids.
    for j in tqdm(range(len(grid_maker.data)), desc=f"task-{tasks[i]}", position=1):
        # pick task from data lodaer, pr: problem, ex: example, desc : id, operation, selcection
        ex_in, ex_out, pr_in, pr_out, desc = grid_maker.pick(data_index=j)

        # data shape
        data = {
            # "desc": {"id": desc['id']+f"_{j}"},
            "desc": {"id": desc['id']},
            "step": [],
            "selection": [],
            "operation": [],
            "operation_name": [],
            "reward": [],
            "terminated": [],
            "grid_dim": [],
            "in_grid": [],
            "out_grid": [],
            "grid": [],
            "clip_dim": [],
            "clip": [],
            "selection_mask": [],
            # input_output examples are added
            "ex_in": [],
            "ex_out": [],
            "ex_in_grid_dim": [],
            "ex_out_grid_dim": []
        }

        # reset environment with loading j-th problem grid.
        obs, info = env.reset(options={'prob_index': j, 'subprob_index': 0, 'adaptation' : False})

        whole_operations = desc['operations']
        whole_selections = desc['selections']

        # initiate values
        grid = np.full(max_grid_dim, 10, dtype=np.uint8)  # For some models, they must have same grid dim. So, fill the outside of real grid with new value '10'.
        sel = [0, 0, 0, 0]  # x,y,h,w (row,col,height,width)
        sel_mask = np.zeros(max_grid_dim, dtype=np.int8)
        reward = 0
        term = False

        g_h, g_w = obs['grid_dim']
        grid_pad = grid.copy()
        grid_pad[:g_h, :g_w] = obs['grid'][:g_h, :g_w]

        c_h, c_w = obs['clip_dim']
        clip_pad = grid.copy()
        clip_pad[:c_h, :c_w] = obs['clip'][:c_h, :c_w]

        for ei, eo in zip(ex_in, ex_out):
            utils.append_example(data, ei, eo, grid)

        time.sleep(0) if render_mode == None else time.sleep(1)

        for s in range(len(whole_operations)):

            time.sleep(0) if render_mode == None else time.sleep(1)

            try:
                operation = whole_operations[s]
                selection = whole_selections[s]
                selection_mask = utils.sel_bbox_to_mask(
                    selection, max_grid_dim)
                action = {'selection': selection_mask.astype(bool), 'operation': operation}
                obs, reward, term, trunc, info = env.step(action)
                utils.append_data(data, grid_pad, selection, g_h, g_w, clip_pad, c_h, c_w, selection_mask, operation, reward, term, s)

                g_h, g_w = obs['grid_dim']
                grid_pad = grid.copy()
                grid_pad[:g_h, :g_w] = obs['grid'][:g_h, :g_w]

                c_h, c_w = obs['clip_dim']
                clip_pad = grid.copy()
                clip_pad[:c_h, :c_w] = obs['clip'][:c_h, :c_w]

            except Exception as e:
                print(f" {tasks[i]} : something wrong in trace! skip this problem")
                # print(e)
                # utils.save_wrong(data, tasks[i], data_folder_path)
                break

        # check it is reasonable trajectory(compare output from ARCLE with handcrafted answer)
        # if not np.array_equal(np.array(data['grid'][-1])[:g_h, :g_w], pr_out[0]):
        if 'expert' in data['desc'] and not np.array_equal(np.array(data['grid'][-1])[:g_h, :g_w], pr_out[0]):
            print(f" {tasks[i]} not correct answer")
            utils.save_wrong(data, tasks[i], data_folder_path)
            continue

        data['in_grid'] = data['grid'][0]
        data['out_grid'] = data['grid'][-1]

        # fit type to save json file
        data = utils.convert_npint_to_int(data)

        if save_whole_trace == "true":
            utils.save_whole(data, tasks[i], data_folder_path)

        # if save option is true, save the sample traces.
        if save_seg_trace == "true":
            # add padding behind of trace for some models. if it is unecessary, delete this part
            sel = [0, 0, 0, 0]
            sel_mask = np.zeros(max_grid_dim, dtype=np.int8)
            # repeat end state.
            for _ in range(H-1):
                utils.append_data(data, grid.copy(), sel, 0, 0, grid.copy(), 0, 0, sel_mask, 35, 0, True, len(data['step']))
                # utils.append_data(data, grid_pad, selection, g_h, g_w, clip_pad, c_h, c_w, selection_mask, operation, reward, term, len(data['step']))

            utils.save_seg(data, tasks[i], H, data_folder_path)

    env.close()