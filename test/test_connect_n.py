import numpy as np

from game import ConnectNGame, Player


def test_win_at_pos():
    n_rows, n_cols, n_to_win = 6, 7, 4
    g = ConnectNGame(n_rows, n_cols, n_to_win)

    # Player 1
    players = [Player.FIRST_PLAYER, Player.SECOND_PLAYER]

    for player in players:
        # No Win
        for r_idx in range(n_rows):
            for c_idx in range(n_cols):
                assert not g.win_at_pos(g.initial_state(), r_idx, c_idx, player)

        # Vertical
        for r_idx in range(n_rows - n_to_win + 1):
            for c_idx in range(n_cols):
                state = g.initial_state()
                state[r_idx: r_idx+n_to_win, c_idx] = player.value

                for r in range(n_rows):
                    for c in range(n_cols):
                        if state[r, c] == player.value:
                            assert g.win_at_pos(state, r, c, player)
                        else:
                            assert not g.win_at_pos(state, r, c, player)

        # Horizontal
        for r_idx in range(n_rows):
            for c_idx in range(n_cols - n_to_win + 1):
                state = g.initial_state()
                state[r_idx, c_idx: c_idx + n_to_win] = player.value

                for r in range(n_rows):
                    for c in range(n_cols):
                        if state[r, c] == player.value:
                            assert g.win_at_pos(state, r, c, player)
                        else:
                            assert not g.win_at_pos(state, r, c, player)

        # Diagonal
        for r_idx in range(n_rows - n_to_win + 1):
            for c_idx in range(n_cols - n_to_win + 1):
                state = g.initial_state()
                for o in range(n_to_win):
                    state[r_idx+o, c_idx+o] = player.value

                for r in range(n_rows):
                    for c in range(n_cols):
                        if state[r, c] == player.value:
                            assert g.win_at_pos(state, r, c, player)
                        else:
                            assert not g.win_at_pos(state, r, c, player)

        for r_idx in range(n_to_win - 1, n_rows):
            for c_idx in range(n_cols - n_to_win + 1):
                state = g.initial_state()
                for o in range(n_to_win):
                    state[r_idx-o, c_idx+o] = player.value

                for r in range(n_rows):
                    for c in range(n_cols):
                        if state[r, c] == player.value:
                            assert g.win_at_pos(state, r, c, player)
                        else:
                            assert not g.win_at_pos(state, r, c, player)
