import numpy as np

from game import ConnectNGame, Player


def test_would_win():
    n_rows, n_cols, n_to_win = 6, 7, 4
    g = ConnectNGame(n_rows, n_cols, n_to_win)

    # Player 1
    players = [Player.FIRST_PLAYER, Player.SECOND_PLAYER]

    for player in players:
        # No Win
        for r_idx in range(n_rows):
            for c_idx in range(n_cols):
                assert not g.would_win(g.initial_state(), r_idx, c_idx, player)

        # Vertical
        for r_idx in range(n_rows - n_to_win + 1):
            for c_idx in range(n_cols):
                state = g.initial_state()
                state[r_idx: r_idx+n_to_win, c_idx] = player.value
                assert g.would_win(state, r_idx, c_idx, player)

        # Horizontal
        for r_idx in range(n_rows):
            for c_idx in range(n_cols - n_to_win + 1):
                state = g.initial_state()
                state[r_idx, c_idx: c_idx + n_to_win] = player.value
                assert g.would_win(state, r_idx, c_idx, player)

        # Diagonal
        for r_idx in range(n_rows - n_to_win + 1):
            for c_idx in range(n_cols - n_to_win + 1):
                pass
                # TODO
                # state = g.initial_state()
                # state[r_idx: r_idx+n_to_win, c_idx] = player.value
                # assert g.would_win(state, r_idx, c_idx, player)
