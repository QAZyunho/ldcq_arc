import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import colors
import shutil
import ffmpeg
import os
import utils
import numpy as np

'''
color map and normalization value to mapping value of grid to each color
'''
cmap = colors.ListedColormap(  # list of colors used in ARC tasks
    [
        '#000000',  # 0 black
        '#0074D9',  # 1 blue
        '#FF4136',  # 2 red
        '#2ECC40',  # 3 green
        '#FFDC00',  # 4 yellow
        '#AAAAAA',  # 5 gray
        '#F012BE',  # 6 pink
        '#FF851B',  # 7 orange
        '#7FDBFF',  # 8 skybkue
        '#870C25',  # 9 maroon
        '#FFFFFF',  # 10 white (for padding)
    ])

norm = colors.Normalize(vmin=0, vmax=10)  # Normalize values to cmap range


def plot_one(mode, is_ex, data, ax, i, seg_id):
    '''
    mode : gif, iniout, segment, whole, wrong
    draw one subplot of whole plot
    is_ex: if data is example or not(problem)
    i: i-th step of trace
    seg_id: seg_id-th separated trace, 0 for ohter modes

    more details are in plot_task part
    '''
    if is_ex:  # set title for example pairs
        input_matrix = data
        if i == 0:
            ax.set_title(f'demonstration input {is_ex}', fontsize=20)
        else:
            ax.set_title(f'demonstration output {is_ex}', fontsize=20)

    else:
        if mode == "inout":  # set title for input-ouput mode
            if i == 0:
                input_matrix = data['in_grid']
                ax.set_title('input')
            else:
                input_matrix = data['out_grid']
                ax.set_title('output')
        else:  # set title for input-ouput mode
            input_matrix = data['grid'][i]
            operation_num = data['operation'][i]
            operation_name = utils.mapping_operation(operation_num)

            if i == 0 and seg_id == 0:  # input title
                ax.set_title(f"test input", fontsize=20)
            else:
                ax.set_title(f"step {seg_id+i}",fontsize=20)
            ax.text(0.5, -0.05, f'{operation_num}  {operation_name}', ha='center', transform=ax.transAxes, fontsize=20)

            # if i == 0:  # input title
            #     if seg_id > 0:
            #         ax.set_title(f"step {seg_id+i}\n{operation_num}  {operation_name}")
            #     else:
            #         ax.set_title(f"input\n{operation_num}  {operation_name}")
            # else:
            #     ax.set_title(f"step {seg_id+i}\n{operation_num}  {operation_name}")

    # mapping data to grid
    ax.pcolormesh(np.flip(input_matrix, 0), cmap=cmap, norm=norm, edgecolors='lightgrey', linewidth=0.05)
    ax.set_aspect('equal')
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    # ax.imshow(input_matrix, cmap=cmap, norm=norm)
    # ax.grid(visible=True, which='both', color='lightgrey', linewidth=0.1)
    # ax.set_yticks([x-0.5 for x in range(0, len(input_matrix)+1)])
    # ax.set_xticks([x-0.5 for x in range(0, len(input_matrix[0])+1)])
    # ax.set_yticks(np.arange(0, len(input_matrix), 5)-0.5, minor=True)
    # ax.set_xticks(np.arange(0, len(input_matrix[0]), 5)-0.5, minor=True)
    # ax.set_xticklabels([])
    # ax.set_yticklabels([])


def save_for_gif(is_ex, data, i, task_id, trace_id, save_folder_path):
    '''
    save each plot as '.png' for making gif 
    is_ex: if data is example or not(problem)
    i: t-th step of trace
    '''

    input_matrix = data['grid'][i]
    operation_num = data['operation'][i]
    operation_name = utils.mapping_operation(operation_num)

    plt.figure(figsize=(5, 5))
    if i == 0:  # input title
        plt.suptitle(f"{task_id}_{trace_id}\ninput", fontsize=30)
    else:  # i-th step title
        plt.suptitle(f"{task_id}_{trace_id}\nstep{i}", fontsize=30)
    plt.text(0.5, -0.1, f'{operation_num}  {operation_name}', ha='center', transform=plt.gca().transAxes,  fontsize=20)

    # mapping data to grid
    plt.pcolormesh(np.flip(input_matrix, 0), cmap=cmap, norm=norm, edgecolors='lightgrey', linewidth=0.01)
    plt.gca().set_aspect('equal')
    plt.gca().xaxis.set_visible(False)
    plt.gca().yaxis.set_visible(False)
    # plt.imshow(input_matrix, cmap=cmap, norm=norm)
    # plt.yticks([x-0.5 for x in range(0, len(input_matrix)+1)])
    # plt.xticks([x-0.5 for x in range(0, len(input_matrix[0])+1)])
    # plt.gca().set_yticks(np.arange(0, len(input_matrix), 5)-0.5, minor=True)
    # plt.gca().set_xticks(np.arange(0, len(input_matrix[0]), 5)-0.5, minor=True)
    # plt.gca().set_xticklabels([])
    # plt.gca().set_yticklabels([])
    # plt.grid(True, which='both', color='lightgrey', linewidth=1.5)
    plt.tight_layout()

    # set file name
    if is_ex:
        if i == 0:
            file_name = f"ex_{is_ex}_in"
        else:
            file_name = f"ex_{is_ex}_out"
    else:
        file_name = f"trace_{i}"

    # making folder part
    if not os.path.exists(save_folder_path):
        os.makedirs(save_folder_path)
    if not os.path.exists(f"{save_folder_path}/{task_id}"):
        os.makedirs(f"{save_folder_path}/{task_id}")
    if not os.path.exists(f"{save_folder_path}/{task_id}/gif"):
        os.makedirs(f"{save_folder_path}/{task_id}/gif")
    if not os.path.exists(f"{save_folder_path}/{task_id}/gif/pngs_{task_id}_{trace_id}"):
        os.makedirs(f"{save_folder_path}/{task_id}/gif/pngs_{task_id}_{trace_id}")

    # save as png
    plt.savefig(f"{save_folder_path}/{task_id}/gif/pngs_{task_id}_{trace_id}/{file_name}.png", bbox_inches='tight', dpi=300)


# def plot_task(mode, data, task_id, trace_id, save_folder_path, make_task_folder=False):
#     '''
#     method for visuallizing trace
#     mode : gif, iniout, segment, whole, wrong
#     data : loaded json data
#     task_id : task_id 
#     trace_id : i-th trace, file name would be {task_id}_{trace_id} or {task_id}_{trace_id}_{segment_id}
#     save_folder_path : folder to save the images
#     '''
#     num_step = len(data['step'])
#     num_examples = len(data['ex_in'])
#     exi = []
#     exo = []
#     axs = []

#     seg_id = 0

#     if mode == "gif":
#         if os.path.exists(f"{save_folder_path}/{task_id}/gif/pngs_{task_id}_{trace_id}"):
#             shutil.rmtree(f"{save_folder_path}/{task_id}/gif/pngs_{task_id}_{trace_id}")

#         for i in range(num_step):
#             save_for_gif(0, data, i, task_id, trace_id, save_folder_path)

#     elif mode == "inout":
#         fig = plt.figure(figsize=(10, 5*(num_examples+1)))
#         gs = GridSpec(nrows=num_examples+1, ncols=2)
#         is_ex = 0

#         for h in range(num_examples):
#             is_ex = h+1
#             exi.append(fig.add_subplot(gs[h, 0]))
#             plot_one(mode, is_ex, data['ex_in'][h], exi[h], 0, seg_id)
#             exo.append(fig.add_subplot(gs[h, 1]))
#             plot_one(mode, is_ex, data['ex_out'][h], exo[h], 1, seg_id)

#         is_ex = 0
#         ax_in = fig.add_subplot(gs[num_examples, 0])
#         ax_out = fig.add_subplot(gs[num_examples, 1])
#         plot_one(mode, is_ex, data, ax_in, 0, seg_id)
#         plot_one(mode, is_ex, data, ax_out, 1, seg_id)

#     else:  # segment // whole // wrong
#         fig = plt.figure(figsize=(5*num_step, 5*(num_examples+1)))
#         # gs = GridSpec(nrows=num_examples+1, ncols=num_step)
#         gs = GridSpec(nrows=num_examples+1, ncols=num_step)
#         is_ex = 0

#         if mode == "segment":
#             seg_id = int(data['desc']['id'].split('.')[0].split('_')[-1])

#         for h in range(num_examples):
#             is_ex = h+1
#             exi.append(fig.add_subplot(gs[h, 0]))
#             plot_one(mode, is_ex, data['ex_in'][h], exi[h], 0, seg_id)
#             exo.append(fig.add_subplot(gs[h, 1]))
#             plot_one(mode, is_ex, data['ex_out'][h], exo[h], 1, seg_id)

#         is_ex = 0

#         for i in range(num_step):
#             axs.append(fig.add_subplot(gs[num_examples, i]))
#             plot_one(mode, is_ex, data, axs[i], i, seg_id)

#     if mode != "gif":  # inout // segment // whole // wrong
#         if 'expert' in data['desc']['id']:
#             title = data['desc']['id'].replace('expert', 'gold-standard')
#         else: 
#             title = data['desc']['id']
#         fig.suptitle(f"{title}, {mode}\n",fontsize=20)

#         if not os.path.exists(save_folder_path):
#             os.makedirs(save_folder_path)
#         if make_task_folder == "true":
#             if not os.path.exists(f"{save_folder_path}/{task_id}"):
#                 os.makedirs(f"{save_folder_path}/{task_id}")
#             if not os.path.exists(f"{save_folder_path}/{task_id}/{mode}"):
#                 os.makedirs(f"{save_folder_path}/{task_id}/{mode}")
#             if mode == 'segment':
#                 if not os.path.exists(f"{save_folder_path}/{task_id}/{mode}/{task_id}_{trace_id}"):
#                     os.makedirs(f"{save_folder_path}/{task_id}/{mode}/{task_id}_{trace_id}")

#             # plt.subplots_adjust(wspace=0.05,hspace=0.3)
#             plt.tight_layout()
#             plt.savefig(f"{save_folder_path}/{task_id}/{mode}/{data['desc']['id']}.png", dpi=300)
#             # plt.show()

#         else:
#             plt.tight_layout()
#             plt.savefig(f"{save_folder_path}/{data['desc']['id']}.png", dpi=300)
#     plt.close()

def plot_task(mode, data, task_id, trace_id, save_folder_path, make_task_folder=False):
    '''
    method for visuallizing trace
    mode : gif, iniout, segment, whole, wrong
    data : loaded json data
    task_id : task_id 
    trace_id : i-th trace, file name would be {task_id}_{trace_id} or {task_id}_{trace_id}_{segment_id}
    save_folder_path : folder to save the images
    '''
    num_step = len(data['step'])
    num_examples = len(data['ex_in'])
    exi = []
    exo = []
    axs = []

    seg_id = 0

    if mode == "gif":
        if os.path.exists(f"{save_folder_path}/{task_id}/gif/pngs_{task_id}_{trace_id}"):
            shutil.rmtree(f"{save_folder_path}/{task_id}/gif/pngs_{task_id}_{trace_id}")

        for i in range(num_step):
            save_for_gif(0, data, i, task_id, trace_id, save_folder_path)

    elif mode == "inout":
        fig = plt.figure(figsize=(10, 5*(num_examples+1)))
        gs = GridSpec(nrows=num_examples+1, ncols=2)
        is_ex = 0

        for h in range(num_examples):
            is_ex = h+1
            exi.append(fig.add_subplot(gs[h, 0]))
            plot_one(mode, is_ex, data['ex_in'][h], exi[h], 0, seg_id)
            exo.append(fig.add_subplot(gs[h, 1]))
            plot_one(mode, is_ex, data['ex_out'][h], exo[h], 1, seg_id)

        is_ex = 0
        ax_in = fig.add_subplot(gs[num_examples, 0])
        ax_out = fig.add_subplot(gs[num_examples, 1])
        plot_one(mode, is_ex, data, ax_in, 0, seg_id)
        plot_one(mode, is_ex, data, ax_out, 1, seg_id)

    else:  # segment // whole // wrong
        fig = plt.figure(figsize=(5*num_step, 5*(num_examples+1)))
        # gs = GridSpec(nrows=num_examples+1, ncols=num_step)
        gs = GridSpec(nrows=num_examples+1, ncols=num_step)
        is_ex = 0

        if mode == "segment":
            seg_id = int(data['desc']['id'].split('.')[0].split('_')[-1])

        for h in range(num_examples):
            is_ex = h+1
            exi.append(fig.add_subplot(gs[h, 0]))
            plot_one(mode, is_ex, data['ex_in'][h], exi[h], 0, seg_id)
            exo.append(fig.add_subplot(gs[h, 1]))
            plot_one(mode, is_ex, data['ex_out'][h], exo[h], 1, seg_id)

        is_ex = 0

        for i in range(num_step):
            axs.append(fig.add_subplot(gs[num_examples, i]))
            plot_one(mode, is_ex, data, axs[i], i, seg_id)

    if mode != "gif":  # inout // segment // whole // wrong
        if 'expert' in data['desc']['id']:
            title = data['desc']['id'].replace('expert', 'gold-standard')
        else: 
            title = data['desc']['id']
        fig.suptitle(f"{title}, {mode}\n",fontsize=20)

        if not os.path.exists(save_folder_path):
            os.makedirs(save_folder_path)
        if make_task_folder == "true":
            if not os.path.exists(f"{save_folder_path}/{task_id}"):
                os.makedirs(f"{save_folder_path}/{task_id}")
            if not os.path.exists(f"{save_folder_path}/{task_id}/{mode}"):
                os.makedirs(f"{save_folder_path}/{task_id}/{mode}")
            
            # For segment mode, save in subdirectory
            if mode == 'segment':
                if not os.path.exists(f"{save_folder_path}/{task_id}/{mode}/{task_id}_{trace_id}"):
                    os.makedirs(f"{save_folder_path}/{task_id}/{mode}/{task_id}_{trace_id}")
                
                plt.tight_layout()
                plt.savefig(f"{save_folder_path}/{task_id}/{mode}/{task_id}_{trace_id}/{data['desc']['id']}.png", dpi=300)
            else:
                plt.tight_layout()
                plt.savefig(f"{save_folder_path}/{task_id}/{mode}/{data['desc']['id']}.png", dpi=300)

        else:
            plt.tight_layout()
            plt.savefig(f"{save_folder_path}/{data['desc']['id']}.png", dpi=300)
    plt.close()

def make_gif(png_folder_path, output_filename):
    '''
    make gif from set of png files.
    fps: fps for output gif
    scale : image size, it can't change the ratio of witdh and height, if do, the image might be cropped.
    for more detail, search about "ffmpeg filter_complex"
    '''
    (
        ffmpeg
        .input(f'{png_folder_path}/trace_%d.png', framerate=2)
        .output(output_filename, filter_complex='[0:v] setsar=1/1,fps=2,scale=w=512:h=-1,split [a][b];[a] palettegen=stats_mode=full [p];[b][p] paletteuse=new=1')
        .run()
    )


def find_data(file_path, path_list=None, json_list=None):
    '''
    find json files from given path
    path_list : path of JSON data file
    json_list : file names of JSON data file
    '''
    if path_list == None:
        path_list = []
    if json_list == None:
        json_list = []

    # If list, iterate the path and frecursively call the function
    if isinstance(file_path, list):
        for path in file_path:
            pl, jl = find_data(path)
            path_list.extend(pl)
            json_list.extend(jl)
    # If directory, iterativley repeat for inner files/directories.
    elif os.path.isdir(file_path):
        for file in os.listdir(file_path):
            # If JSON file, then add to list
            if file.split('.')[-1] == 'json':
                path_list.append(os.path.join(file_path, file))
                json_list.append(file)
            else:
                pls, jls = find_data(os.path.join(file_path, file))
                path_list.extend(pls)
                json_list.extend(jls)
     # If JSON file, then add to list
    elif file_path.split('.')[-1] == 'json':
        path_list.append(file_path)
        json_list.append(file_path.split('/')[-1])

    return path_list, json_list