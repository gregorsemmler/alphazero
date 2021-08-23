import logging
from collections import Counter
from os import listdir
from os.path import join
from timeit import default_timer as timer

import torch

from game import ConnectNGame
from mcts import MonteCarloTreeSearch
from model import CNNModel
from utils import load_checkpoint


logger = logging.getLogger(__name__)


def tournament():
    logging.basicConfig(level=logging.INFO)

    # model_path = "model_checkpoints/best/tournament_23082021_1"
    model_path = "model_checkpoints/best/tournament_22082021_23082021_1"
    model_name_filter = ""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cpu = torch.device("cpu")
    model_names = [fn for fn in listdir(model_path) if model_name_filter in fn]

    n_rows, n_cols, n_to_win = 6, 7, 4
    game = ConnectNGame(n_rows, n_cols, n_to_win)

    num_input_states = 2
    input_shape = (2 * num_input_states, game.n_rows, game.n_cols)
    num_filters = 64
    num_residual_blocks = 3
    val_hidden_size = 20

    num_eval_mcts_searches = 10

    model_d = {}
    wins_d = Counter()
    draws_d = Counter()
    losses_d = Counter()
    score_d = Counter()

    for model_name in model_names:
        m = CNNModel(input_shape, num_filters, num_residual_blocks, val_hidden_size, game.n_cols)
        load_checkpoint(join(model_path, model_name), m)
        model_d[model_name] = m

    num_matches = 3

    for model_name1, m1 in model_d.items():
        for model_name2, m2 in model_d.items():
            if model_name1 == model_name2:
                continue

            m1.to(device)
            m2.to(device)

            mcts1 = MonteCarloTreeSearch(m1, game, num_input_states, device=device)
            mcts2 = MonteCarloTreeSearch(m2, game, num_input_states, device=device)

            n1, n2, n_draws = 0, 0, 0

            s = timer()
            n_steps = 0
            for match_idx in range(num_matches):
                r, m_steps = mcts1.play_match(num_eval_mcts_searches, lambda x: 0.0, other_mcts=mcts2)
                n_steps += m_steps

                if r > 0:
                    n1 += 1
                    wins_d[model_name1] += 1
                    losses_d[model_name2] += 1

                    score_d[model_name1] += 3
                elif r < 0:
                    n2 += 1
                    wins_d[model_name2] += 1
                    losses_d[model_name1] += 1

                    score_d[model_name2] += 3
                else:
                    draws_d[model_name1] += 1
                    draws_d[model_name2] += 1

                    score_d[model_name1] += 1
                    score_d[model_name2] += 1
                    n_draws += 1

            e = timer()
            logger.info(f"{num_matches} matches between {model_name1} and {model_name2} with {n_steps} steps "
                        f"took {e - s:.3f} seconds. Result: {n1}/{n2}/{n_draws}")

            m1.to(cpu)
            m2.to(cpu)

    logger.info("Final Leaderboard:")
    for model_name, score in score_d.most_common():
        n_wins = wins_d[model_name]
        n_losses = losses_d[model_name]
        n_draws = draws_d[model_name]
        logger.info(f"{model_name}:\tScore: {score}\tW/L/D: {n_wins}/{n_losses}/{n_draws}")

    print("")
    pass


if __name__ == "__main__":
    tournament()
