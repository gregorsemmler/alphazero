from enum import Enum

import numpy as np


class Player(Enum):
    FIRST_PLAYER = 0
    SECOND_PLAYER = 1


class Game(object):

    def valid_actions(self, state):
        raise NotImplementedError()

    def invalid_actions(self, state):
        raise NotImplementedError()

    def move(self, state, action, player):
        raise NotImplementedError()

    def is_draw(self, state):
        raise NotImplementedError()

    def render(self, state):
        raise NotImplementedError()


class Connect4Game(Game):

    def __init__(self, num_rows=6, num_cols=7, count_to_win=4):
        super().__init__()
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.count_to_win = count_to_win
        if self.count_to_win > self.num_rows or self.count_to_win > self.num_cols:
            raise ValueError("Board dimensions are too small to allow win.")

    def valid_actions(self, state):
        is_valid = state[0] == 0.0
        return {idx for idx in range(len(is_valid)) if is_valid[idx]}

    def invalid_actions(self, state):
        is_valid = state[0] == 0.0
        return {idx for idx in range(len(is_valid)) if not is_valid[idx]}

    def move(self, state, action, player):
        if action in self.invalid_actions(state):
            raise ValueError("Invalid action.")

        res_state = state.copy()
        column = res_state[:, action]
        row_idx = column[np.where(column == 0.0)][0].max()
        column[row_idx] = player.value
        # TODO
        pass

    def would_win(self, new_state, row_idx, col_idx, player):
        if new_state.shape != (self.num_rows, self.num_cols):
            raise ValueError("State is in incorrect shape")

        # vertical
        if self.num_rows - row_idx > self.count_to_win:
            vert_line: np.ndarray = new_state[row_idx: row_idx+self.count_to_win, col_idx]
            if (vert_line == player.value).all():
                return True

        start_col_idx = max(0, col_idx-self.count_to_win+1)
        end_col_idx = min(self.num_cols-self.count_to_win+1, col_idx+1)
        start_row_idx = max(0, row_idx-self.count_to_win+1)
        end_row_idx = min(self.num_rows-self.count_to_win+1, row_idx+1)

        for c_idx in range(start_col_idx, end_col_idx):
            # horizontal
            row: np.ndarray = new_state[row_idx, c_idx: c_idx+self.count_to_win]
            if (row == player.value).all():
                return True

        # diagonal
        for r_idx, c_idx in zip(range(start_row_idx, end_row_idx), range(start_col_idx, end_col_idx)):
            line1 = [new_state[r_idx+a_idx, c_idx+a_idx] for a_idx in range(self.count_to_win)]
            if all([el == player.value for el in line1]):
                return True

        pass

    def state_to_string(self, state):
        index_line = "".join([str(el) for el in range(self.num_cols)])
        sep_line = "-" * self.num_cols
        lines = [index_line, sep_line]

        def val_to_char(val):
            if val == 1.0:
                return "O"
            if val == -1.0:
                return "X"
            return " "

        for i_row in range(self.num_rows):
            line = "".join([val_to_char(el) for el in state[i_row]])
            lines.append(line)

        lines.extend([index_line, sep_line])
        return "\n".join(lines)

    def render(self, state):
        print(self.state_to_string(state))

