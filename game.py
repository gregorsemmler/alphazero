from enum import Enum

import numpy as np
import torch


class Player(Enum):
    NO_PLAYER = 0
    FIRST_PLAYER = 1
    SECOND_PLAYER = 2


def switch_player(p):
    return Player.FIRST_PLAYER if p == Player.SECOND_PLAYER else Player.SECOND_PLAYER


class Game(object):

    def initial_state(self):
        raise NotImplementedError()

    def valid_actions(self, state):
        raise NotImplementedError()

    def invalid_actions(self, state):
        raise NotImplementedError()

    @property
    def num_actions(self):
        raise NotImplementedError()

    def move(self, state, action, player):
        raise NotImplementedError()

    def state_to_tensor(self, state, device):
        raise NotImplementedError()

    def render(self, state):
        raise NotImplementedError()

    def encode_state(self, state):
        raise NotImplementedError()


class ConnectNGame(Game):

    def __init__(self, n_rows=6, n_cols=7, count_to_win=4):
        super().__init__()
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_actions = n_cols
        self.count_to_win = count_to_win
        if self.count_to_win > self.n_rows or self.count_to_win > self.n_cols:
            raise ValueError("Board dimensions are too small to allow win.")

    def initial_state(self):
        s = np.zeros((self.n_rows, self.n_cols))
        s[:, :] = Player.NO_PLAYER.value
        return s

    def valid_actions(self, state):
        is_valid = state[0] == Player.NO_PLAYER.value
        return {idx for idx in range(len(is_valid)) if is_valid[idx]}

    def invalid_actions(self, state):
        is_valid = state[0] == Player.NO_PLAYER.value
        return {idx for idx in range(len(is_valid)) if not is_valid[idx]}

    def encode_state(self, state):
        return hash(state.data.tobytes())

    @property
    def num_actions(self):
        return self.n_actions

    def move(self, state, action, player):
        if action in self.invalid_actions(state):
            raise ValueError("Invalid action.")

        res_state = state.copy()
        column = res_state[:, action]
        row_idx = np.where(column == Player.NO_PLAYER.value)[0].max()
        column[row_idx] = player.value

        return res_state, self.win_at_pos(res_state, row_idx, action, player)

    def win_at_pos(self, ar, row_idx, col_idx, player):
        if ar.shape != (self.n_rows, self.n_cols):
            raise ValueError("State is in incorrect shape")

        # vertical
        start_row_idx = max(0, row_idx - self.count_to_win + 1)
        end_row_idx = min(self.n_rows - self.count_to_win + 1, row_idx + 1)

        for r_idx in range(start_row_idx, end_row_idx):
            vert_line: np.ndarray = ar[r_idx: r_idx + self.count_to_win, col_idx]
            if (vert_line == player.value).all():
                return True

        start_col_idx = max(0, col_idx-self.count_to_win+1)
        end_col_idx = min(self.n_cols - self.count_to_win + 1, col_idx + 1)
        for c_idx in range(start_col_idx, end_col_idx):
            # horizontal
            row: np.ndarray = ar[row_idx, c_idx: c_idx + self.count_to_win]
            if (row == player.value).all():
                return True

        # Diagonal 1
        move_upper = min(row_idx, col_idx, self.count_to_win - 1)
        move_lower = min(self.n_rows - row_idx - 1, self.n_cols - col_idx - 1, self.count_to_win - 1)

        for off1 in range(-1 * move_upper, move_lower - (self.count_to_win - 1) + 1):
            d = np.array([ar[row_idx + off1 + i, col_idx + off1 + i] for i in range(self.count_to_win)])
            if (d == player.value).all():
                return True

        # Diagonal 2
        move_upper2 = min(self.n_cols - col_idx - 1, row_idx, self.count_to_win - 1)
        move_lower2 = min(col_idx, self.n_rows - row_idx - 1, self.count_to_win - 1)

        for off2 in range(-1 * move_lower2, move_upper2 - (self.count_to_win - 1) + 1):
            d = np.array([ar[row_idx + (-1) * off2 + (-1) * i, col_idx + off2 + i] for i in range(self.count_to_win)])
            if (d == player.value).all():
                return True

        return False

    @staticmethod
    def val_to_char(val):
        if val == Player.FIRST_PLAYER.value:
            return "O"
        if val == Player.SECOND_PLAYER.value:
            return "X"
        return " "

    def state_to_string(self, state):
        index_line = "".join([str(el) for el in range(self.n_cols)])
        sep_line = "-" * self.n_cols
        lines = [index_line, sep_line]

        for i_row in range(self.n_rows):
            line = "".join([self.val_to_char(el) for el in state[i_row]])
            lines.append(line)

        lines.extend([index_line, sep_line])
        return "\n".join(lines)

    def state_to_tensor(self, state, device):
        result = torch.zeros(2, self.n_rows, self.n_cols, device=device)
        for r_idx in range(self.n_rows):
            for c_idx in range(self.n_cols):
                if state[r_idx, c_idx] == Player.FIRST_PLAYER.value:
                    result[0, r_idx, c_idx] = 1
                elif state[r_idx, c_idx] == Player.SECOND_PLAYER.value:
                    result[1, r_idx, c_idx] = 1
        return result

    def render(self, state):
        print(self.state_to_string(state))

