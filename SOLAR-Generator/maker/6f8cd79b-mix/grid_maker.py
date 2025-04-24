from maker.base_grid_maker import BaseGridMaker
from typing import Dict, List, Tuple
from numpy.typing import NDArray
import numpy as np


class GridMaker(BaseGridMaker):

    def parse(self, **kwargs) -> List[Tuple[List[NDArray], List[NDArray], List[NDArray], List[NDArray], Dict]]:
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

            operations = []
            selections = []

            # 공통

            # 색상
            l_color = np.random.randint(1, 10)

            # 두께
            l_thick = 1

            # 내 구현
            j = 0
            # num_examples 만큼 예제 만들기 + 마지막 1개 test case
            while (j < num_examples+1):
                h = np.random.randint(2 * l_thick + 1, max_h)
                w = np.random.randint(2 * l_thick + 1, max_w)
                rand_grid = np.zeros((h, w), dtype=np.uint8)

                # answer 만들때 조심! 세개 다 같음
                answer_grid = rand_grid.copy()

                # 선 긋기
                answer_grid[0:l_thick, :] = l_color
                answer_grid[h-l_thick:, :] = l_color
                answer_grid[:, :l_thick] = l_color
                answer_grid[:, w-l_thick:] = l_color

                # ARCLE
                if (j == num_examples):
                    choice = np.random.randint(3)
                    if choice == 0:     # ㅣ,ㅣ,ㅡ,ㅡ
                        selections.append([0, 0, h - 1, l_thick - 1])               # ㅣ 색칠
                        operations.append(l_color)    # Color
                        selections.append([0, w - l_thick, h - 1, l_thick - 1])     # ㅣ 색칠
                        operations.append(l_color)    # Color
                        selections.append([0, 0, l_thick - 1, w - 1])               # ㅡ 색칠
                        operations.append(l_color)    # Color
                        selections.append([h - l_thick, 0, l_thick - 1, w - 1])     # ㅡ 색칠
                        operations.append(l_color)    # Color

                    elif choice == 1:
                        selections.append([0, 0, h - 1, l_thick - 1])               # ㅣ 색칠
                        operations.append(l_color)    # Color
                        selections.append([0, w - l_thick, h - 1, l_thick - 1])     # ㅣ 색칠
                        operations.append(l_color)    # Color
                        selections.append([0, l_thick, l_thick - 1, w - 1 - 2 * l_thick])               # ㅡ 색칠
                        operations.append(l_color)    # Color
                        selections.append([h - l_thick, l_thick, l_thick - 1, w - 1 - 2 * l_thick])     # ㅡ 색칠
                        operations.append(l_color)    # Color

                    elif choice == 2:
                        selections.append([l_thick, 0, h - l_thick - 1, l_thick - 1])               # ㅣ 색칠
                        operations.append(l_color)    # Color
                        selections.append([l_thick, w - l_thick, h - l_thick - 1, l_thick - 1])     # ㅣ 색칠
                        operations.append(l_color)    # Color
                        selections.append([0, 0, l_thick - 1, w - 1])               # ㅡ 색칠
                        operations.append(l_color)    # Color
                        selections.append([h - l_thick, 0, l_thick - 1, w - 1])     # ㅡ 색칠
                        operations.append(l_color)    # Color

                    operations.append(34)   # Submit
                    selections.append([0, 0, h-1, w-1])

                    pr_in.append(rand_grid)
                    pr_out.append(answer_grid)
                    j = j + 1

                # Example case 저장
                else:
                    ex_in.append(rand_grid)
                    ex_out.append(answer_grid)
                    j = j + 1

            desc = {'id': '6f8cd79b',
                    'selections': selections,
                    'operations': operations}
            dat.append((ex_in, ex_out, pr_in, pr_out, desc))
            for i in range(9):
                branch_idx = np.random.randint(1, len(operations) - 2)  # Submit 전까지만 고려

                new_operations = operations[:branch_idx]  # 앞부분은 expert
                new_selections = selections[:branch_idx]  # selections는 동일

                # 하나의 랜덤 색상 선택
                random_color = np.random.randint(1, 10)

                # branch 이후 selection 수만큼 동일한 색상으로 채움
                num_remaining = len(operations) - 1 - branch_idx
                new_operations += [random_color] * num_remaining

                # Submit 추가
                new_operations.append(34)
                new_selections = selections[:len(new_operations)]

                desc = {
                    'id': f'6f8cd79b-random_{num}_{i}',
                    'selections': new_selections,
                    'operations': new_operations,
                }
                dat.append((ex_in, ex_out, pr_in, pr_out, desc))
        return dat