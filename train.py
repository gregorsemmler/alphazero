import logging

import torch

from game import ConnectNGame
from model import CNNModel


def main():
    logging.basicConfig(level=logging.INFO)

    device_token = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_token)

    game = ConnectNGame()

    input_shape = (2, game.n_rows, game.n_cols)
    num_filters = 64
    num_residual_blocks = 5
    val_hidden_size = 20
    model = CNNModel(input_shape, num_filters, num_residual_blocks, val_hidden_size, game.n_cols)

    pass


if __name__ == "__main__":
    main()
