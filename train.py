import logging
from collections import deque
from copy import deepcopy

import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR

from game import ConnectNGame
from mcts import MCTS
from model import CNNModel


def main():
    logging.basicConfig(level=logging.INFO)

    device_token = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_token)

    n_rows, n_cols, n_to_win = 6, 7, 4
    game = ConnectNGame(n_rows, n_cols, n_to_win)

    input_shape = (2, game.n_rows, game.n_cols)
    num_filters = 64
    num_residual_blocks = 5
    val_hidden_size = 20
    model = CNNModel(input_shape, num_filters, num_residual_blocks, val_hidden_size, game.n_cols).to(device)
    best_model = deepcopy(model)
    mcts = MCTS(model, game, game.n_actions, device_token=device_token)

    replay_buffer_size = 100000
    lr = 0.1
    momentum = 0.9
    l2_regularization = 1e-4
    train_steps = 20  # TODO after how many batch updates to save a checkpoint and evaluate with best model
    min_size_to_train = 2000

    def simple_tau_sched(x):
        return 0 if x > 30 else 1

    num_mcts_searches = 100
    num_games_played = 50
    milestones = [int(el) for el in [200e3, 400e3, 600e3]]  # Milestones for mini-batch lr scheduling steps from paper

    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=l2_regularization)
    scheduler = MultiStepLR(optimizer, milestones=milestones)
    replay_buffer = deque(maxlen=replay_buffer_size)

    step_idx = 0
    train_idx = 0

    while True:

        for game_idx in range(num_games_played):
            _, match_steps = mcts.play_match(num_mcts_searches, simple_tau_sched, replay_buffer=replay_buffer)

        if len(replay_buffer) < min_size_to_train:
            continue

        for batch_idx in range(train_steps):
            pass


        pass
    pass


if __name__ == "__main__":
    main()
