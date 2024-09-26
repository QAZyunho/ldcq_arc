from arcle.loaders import Loader
from typing import Dict, List, Tuple
from numpy.typing import NDArray
import numpy as np
import random


class GridMaker(Loader):

    def parse(self, **kwargs) -> List[Tuple[List[NDArray], List[NDArray], List[NDArray], List[NDArray], Dict]]:
        '''
        #TODO
        create random grids.
        three main steps:
            1. Set common rules
            2. Make input-output grid pairs, operation, selection
            3. Formatting data for passing to ARCLE.

        pr_in: problem input for trace (saved in the list but only one trace), matched with 'ei' in ARCLE
        pr_out: problem output for trace (saved in the list but only one trace), matched with 'eo' in ARCLE
        ex_in: list of example input, matched with 'ti' in ARCLE
        ex_out: list of example output, matched with 'to' in ARCLE
        selections: list of [x,y,h,w] pairs for each operation
        operations: list of integer operations used in ARCLE
        data will be stored at dat[(ex_in,ex_out,pr_in,pr_out,desc)]

        arguments
        num_samples: number of samples to generate for each task
        max_h, max_w: maximum grid dim height, width
        num_examples: number of example pairs for each trace data

        #1~3 is just an example for are for illustrative purposes. Thus You can modify #1~3 parts free. There is no limitation if you return the correct 'dat'
        '''

        dat = []
        num = 0

        num_samples = kwargs['num_samples']
        max_h, max_w = kwargs['max_grid_dim']
        num_examples = kwargs['num_examples']

        while num < num_samples:
            num += 1
            pr_in: List[NDArray] = []
            pr_out: List[NDArray] = []
            ex_in: List[NDArray] = []
            ex_out: List[NDArray] = []

            operations = []
            selections = []

            # 1. Common part
            '''
            In this part, you can define rules that are common to both the example pairs and the given trace. 
            The rule represents what should not be changed in the data. 
            For example, in example pairs, some operation pattern[rotate, flip, rotate, flip] can be a key point of this data, however, if you choose another pattern [flip, flip, flip, flip] is the wrong way to generate the problem.
            You can set several rules within a task, so any rule can be applied in a task as long as there is consistency in a data structure '(ex_in,ex_out,pr_in,pr_out,desc)'. 
             
            What you can decide: Color for objects, Color for making some pattern grid, operation patterns, ...
            '''
            # Just example, it might be unecessary to your code.
            color_list = random.sample(range(1, 10), 2)
            possible_list = [24, 25, 26, 27]
            opr_list = [possible_list[np.random.randint(4)] for i in range(4)]
            flip_direction = np.random.randint(2)

            # 2. Data creating part
            j = 0
            while (j < num_examples+1):

                '''
                You have to make a trace to get the correct answer.
                    2-1. create grid
                    2-2. make proper trace according to the rules.


                h: height of random grid
                w: width of random grid

                In each step, you have to clarify opr, x,y,sel_h,sel_w.
                Think about where to select to apply the operation properly.

                opr: operation number(check 'action_names' in utils.py)  # 0~35
                x: starting height(row) of selection
                y: starting width(column) of selection
                sel_h: number of selected unit squares on the height axis except for itself. ex) 0 means just select itself on the height axis.
                sel_w: number of selected unit squares on the width axis except for itself. ex) 0 means just select itself on the width axis.
                operations[i] must be matched with selections[i]
                '''

                # 2-1. generate grid
                h = np.random.randint(1, max_h+1)
                w = np.random.randint(1, max_w+1)
                input_grid = np.random.randint(0, 10, size=[h, w])

                # 2-2. add operation / selection
                output_grid = input_grid
                for opr in opr_list:
                    output_grid = some_action(output_grid)
                    x, y, sel_h, sel_w = 0, 0, 0, 0
                    operations.append(opr)  # 0~35
                    selections.append([x, y, sel_h, sel_w])

                # 3. Formatting data for passing to ARCLE.
                # Trace case
                if (j == num_examples):
                    # In ARCLE, the 'Submit' action confirms whether the answer is correct or not and gives a reward.
                    operations.append(34)   # Submit.
                    selections.append([0, 0, h-1, w-1])

                    pr_in.append(input_grid)
                    pr_out.append(output_grid)
                    j = j + 1

                # Example case
                else:
                    ex_in.append(input_grid)
                    ex_out.append(output_grid)
                    j = j + 1

            desc = {'id': 'task_id',  # ***important*** task_id as atring
                    'selections': selections,  # sequence of [x,y,sel_h, sel_w] pairs
                    'operations': operations}  # sequence of operations

            dat.append((ex_in, ex_out, pr_in, pr_out, desc))

        return dat


'''
If you need, you can define several methods for making answer or sequence of operation and selection pair.
If not, you can code only in the 'parse' method.
'''


def make_answer(self, input_grid):
    raise NotImplementedError


def some_action(self, input_grid):
    raise NotImplementedError
