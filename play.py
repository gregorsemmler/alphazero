from os.path import basename

import numpy as np
import torch
import torch.nn.functional as F

from game import ConnectNGame, Player, switch_player
from model import CNNModel
from utils import load_checkpoint


def play_against_model():
    model_path = "model_checkpoints/best/testrun1_26072021_100008_best_93.tar"
    model_id = basename(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_rows, n_cols, n_to_win = 6, 7, 4
    game = ConnectNGame(n_rows, n_cols, n_to_win)

    input_shape = (2, game.n_rows, game.n_cols)
    num_filters = 64
    num_residual_blocks = 3
    val_hidden_size = 20

    model = CNNModel(input_shape, num_filters, num_residual_blocks, val_hidden_size, game.n_cols)
    load_checkpoint(model_path, model, device=device)
    model.eval()

    print(f"Playing against {model_id}")
    print(f"Pick a color, {ConnectNGame.val_to_char(Player.FIRST_PLAYER.value)}={Player.FIRST_PLAYER.value}, "
          f"{ConnectNGame.val_to_char(Player.SECOND_PLAYER.value)}={Player.SECOND_PLAYER.value}")

    human_player = Player[input()]
    model_player = switch_player(human_player)
    whose_turn = np.random.choice([human_player, model_player])

    while True:
        state = game.initial_state()
        game_over = False

        while not game_over:
            game.render(state)
            if whose_turn == human_player:
                print(f"Your Turn (Pick a value between 0 and {game.num_actions})")
                human_action_taken = False

                human_action = None
                while not human_action_taken:
                    human_action = int(input())
                    if human_action in game.invalid_actions(state):
                        print("This is not a valid action.")
                    else:
                        human_action_taken = True

                state, won = game.move(state, human_action, human_player)

            else:
                # Play without MCTS here
                print("My Turn")
                model_in = game.state_to_tensor(state, device=device).unsqueeze(0)

                with torch.no_grad():
                    log_probs_out, values_out = model(model_in)
                    prior_probs_out = F.softmax(log_probs_out, dim=1)

                values_np = values_out.detach().cpu().numpy()
                prior_probs_np = prior_probs_out.detach().cpu().numpy()

                print("State")


    pass


if __name__ == "__main__":
    play_against_model()
