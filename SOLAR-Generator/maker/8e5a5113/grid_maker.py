from typing import Dict, List, Tuple
from numpy.typing import NDArray
import numpy as np
from maker.base_grid_maker import BaseGridMaker


class GridMaker(BaseGridMaker):

    def parse(self, **kwargs) -> List[Tuple[List[NDArray], List[NDArray], List[NDArray], List[NDArray], Dict, List[NDArray], List[NDArray]]]:
        dat = []
        num = 0

        num_samples = kwargs['num_samples']
        max_h, max_w = kwargs['max_grid_dim']
        num_examples = kwargs['num_examples']

        # randomly generate inputs
        while num < num_samples:
            num += 1
            pr_in: List[NDArray] = []
            pr_out: List[NDArray] = []
            ex_in: List[NDArray] = []
            ex_out: List[NDArray] = []
            selections: List[NDArray] = []
            operations: List[NDArray] = []

            # 0 for horizontal 1 for vetical
            stack_direction = np.random.randint(2)

            # start from left or right 0: right 1: left
            start_left = np.random.randint(2)

            # number of small grid
            num_small_grid = np.random.randint(2, max_h//2)

            # width of wall
            d = np.random.randint(
                1, (max_h-num_small_grid)//(num_small_grid-1)+1)

            # possible action list rotate/ flip
            possible_list = [24, 25, 26, 27]

            # select random actions for small grid
            action_list = np.random.choice(possible_list, size=num_small_grid-1)

            # select wall_color and other(pr_out make small grids)
            wall_color = np.random.randint(1, 10)

            for _ in range(num_examples):
                h = np.random.randint(
                    1, (max_h-d*(num_small_grid-1))//(num_small_grid)+1)
                w = h
                if h == 1:
                    other_color_list = [
                        i for i in range(1, 10) if i != wall_color]
                else:
                    other_color_list = [
                        i for i in range(10) if i != wall_color]
                rand_grid = np.random.choice(
                    other_color_list, size=[h, w])
                in_grid = self.make_grid(
                    rand_grid, h, w, start_left, num_small_grid, d, wall_color, stack_direction)
                ex_in.append(in_grid.copy())
                ex_out.append(self.make_answer_grid(in_grid, h, w, start_left, d, operations, selections, action_list, stack_direction, False))

            # set grid size randomly and randomly generate grid.
            # 0 for horizontal 1 for vetical 2 for both(Rectangular)
            h = np.random.randint(
                1, (max_h-(num_small_grid-1)*d)//(num_small_grid)+1)
            w = h
            if h == 1:
                other_color_list = [
                    i for i in range(1, 10) if i != wall_color]
            else:
                other_color_list = [
                    i for i in range(10) if i != wall_color]
            rand_grid = np.random.choice(other_color_list, size=[h, w])

            in_grid = self.make_grid(
                rand_grid, h, w, start_left, num_small_grid, d, wall_color, stack_direction)

            pr_in.append(in_grid.copy())
            pr_out.append(self.make_answer_grid(in_grid, h, w, start_left, d, operations, selections, action_list, stack_direction, True))

            desc = {'id': '8e5a5113',
                    'selections': selections,
                    'operations': operations}

            dat.append((ex_in, ex_out, pr_in, pr_out, desc))

        return dat

    def make_grid(self, rand_grid, h, w, start_left, num_small_grid, d, wall_color, stack_direction):
        if stack_direction == 1:
            in_h = (h+d)*num_small_grid-d
            in_w = w
            wall_grid = np.full((d, w), wall_color)
        else:
            in_h = h
            in_w = (w+d)*num_small_grid-d
            wall_grid = np.full((h, d), wall_color)

        in_grid = np.zeros((in_h, in_w))

        if start_left:
            in_grid[:h, :w] = rand_grid.copy()
        else:
            if stack_direction == 1:
                in_grid[in_h-h:in_h, :in_w] = rand_grid.copy()
            else:
                in_grid[:in_h, in_w - w:in_w] = rand_grid.copy()

        for i in range(1, num_small_grid):
            if stack_direction == 1:
                in_grid[h*i+d*i-d:h*i+d*i, :in_w] = wall_grid.copy()
            else:
                in_grid[:in_h, w*i+d*i-d:w*i+d*i] = wall_grid.copy()

        return in_grid

    def make_answer_grid(self, in_grid, h, w, start_left, d, operations, selections, action_list, stack_direction, for_test):
        out_grid = in_grid.copy()

        if stack_direction == 1:  # stack vertical
            if start_left:  # start from left(top)
                for i, opr in enumerate(action_list):
                    low_h = h*i + d*i
                    high_h = h*(i+1) + d*i

                    if h == 1:
                        out_grid[low_h+h+d:high_h+h+d, :w] = out_grid[low_h:high_h, :w].copy()

                    elif opr == 24:
                        out_grid[low_h+h+d:high_h+h+d, :w] = np.rot90(out_grid[low_h:high_h, :w]).copy()
                    elif opr == 25:
                        out_grid[low_h+h+d:high_h+h+d, :w] = np.rot90(out_grid[low_h:high_h, :w], 3).copy()
                    elif opr == 26:
                        out_grid[low_h+h+d:high_h+h+d, :w] = np.flip(out_grid[low_h:high_h, :w], 1).copy()
                    elif opr == 27:
                        out_grid[low_h+h+d:high_h+h+d, :w] = np.flip(out_grid[low_h:high_h, :w], 0).copy()
                    else:
                        raise NotImplementedError

                    if for_test:
                        sel = [h*i+d*i, 0, h-1, w-1]
                        selections.append(sel.copy())
                        operations.append(29)

                        sel = [h*(i+1)+d*(i+1), 0, h-1, w-1]
                        selections.append(sel.copy())
                        operations.append(30)

                        if h != 1:
                            selections.append(sel.copy())
                            operations.append(opr)

            else:  # start from right(bottom)
                for i, opr in enumerate(action_list):
                    l = len(action_list)
                    low_h = h*(l-i) + d*(l-i)
                    high_h = h*(l-i+1) + d*(l-i)

                    if h == 1:
                        out_grid[low_h-h-d:high_h-h-d, :w] = out_grid[low_h:high_h, :w].copy()

                    elif opr == 24:
                        out_grid[low_h-h-d:high_h-h-d, :w] = np.rot90(out_grid[low_h:high_h, :w]).copy()
                    elif opr == 25:
                        out_grid[low_h-h-d:high_h-h-d, :w] = np.rot90(out_grid[low_h:high_h, :w], 3).copy()
                    elif opr == 26:
                        out_grid[low_h-h-d:high_h-h-d, :w] = np.flip(out_grid[low_h:high_h, :w], 1).copy()
                    elif opr == 27:
                        out_grid[low_h-h-d:high_h-h-d, :w] = np.flip(out_grid[low_h:high_h, :w], 0).copy()
                    else:
                        raise NotImplementedError

                    if for_test:
                        sel = [h*(l-i)+d*(l-i), 0, h-1, w-1]
                        selections.append(sel.copy())
                        operations.append(29)

                        sel = [h*(l-i-1)+d*(l-i-1), 0, h-1, w-1]
                        selections.append(sel.copy())
                        operations.append(30)

                        if h != 1:
                            selections.append(sel.copy())
                            operations.append(opr)

        else:  # stack horizontal
            if start_left:  # start from left
                for i, opr in enumerate(action_list):
                    low_w = w*i + d*i
                    high_w = w*(i+1) + d*i

                    if h == 1:
                        out_grid[:h, low_w+w+d:high_w+w+d] = out_grid[:h, low_w:high_w].copy()

                    elif opr == 24:
                        out_grid[:h, low_w+w+d:high_w+w+d] = np.rot90(out_grid[:h, low_w:high_w]).copy()
                    elif opr == 25:
                        out_grid[:h, low_w+w+d:high_w+w+d] = np.rot90(out_grid[:h, low_w:high_w], 3).copy()
                    elif opr == 26:
                        out_grid[:h, low_w+w+d:high_w+w+d] = np.flip(out_grid[:h, low_w:high_w], 1).copy()
                    elif opr == 27:
                        out_grid[:h, low_w+w+d:high_w+w+d] = np.flip(out_grid[:h, low_w:high_w], 0).copy()
                    else:
                        raise NotImplementedError

                    if for_test:
                        sel = [0, w*i+d*i, h-1, w-1]
                        selections.append(sel.copy())
                        operations.append(29)

                        sel = [0, w*(i+1)+d*(i+1), h-1, w-1]
                        selections.append(sel.copy())
                        operations.append(30)

                        if h != 1:
                            selections.append(sel.copy())
                            operations.append(opr)

            else:  # start from right
                for i, opr in enumerate(action_list):
                    l = len(action_list)
                    low_w = w*(l-i) + d*(l-i)
                    high_w = w*(l-i+1) + d*(l-i)

                    if h == 1:
                        out_grid[:h, low_w-w-d:high_w-w-d] = out_grid[:h, low_w:high_w].copy()

                    elif opr == 24:
                        out_grid[:h, low_w-w-d:high_w-w-d] = np.rot90(out_grid[:h, low_w:high_w]).copy()
                    elif opr == 25:
                        out_grid[:h, low_w-w-d:high_w-w-d] = np.rot90(out_grid[:h, low_w:high_w], 3).copy()
                    elif opr == 26:
                        out_grid[:h, low_w-w-d:high_w-w-d] = np.flip(out_grid[:h, low_w:high_w], 1).copy()
                    elif opr == 27:
                        out_grid[:h, low_w-w-d:high_w-w-d] = np.flip(out_grid[:h, low_w:high_w], 0).copy()
                    else:
                        raise NotImplementedError

                    if for_test:
                        sel = [0, w*(l-i)+d*(l-i), h-1, w-1]
                        selections.append(sel.copy())
                        operations.append(29)

                        sel = [0, w*(l-i-1)+d*(l-i-1), h-1, w-1]
                        selections.append(sel.copy())
                        operations.append(30)

                        if h != 1:
                            selections.append(sel.copy())
                            operations.append(opr)
        if for_test:
            out_h, out_w = in_grid.shape
            selections.append([0, 0, out_h-1, out_w-1])
            operations.append(34)

        return out_grid
