from os.path import basename

import numpy as np
import torch
import torch.nn.functional as F

from game import ConnectNGame, Player, switch_player
from mcts import MonteCarloTreeSearch
from model import CNNModel
from utils import load_checkpoint


def predict_with_model(game, model, state, device, show_hints=True):
    model_in = game.state_to_tensor(state, device=device).unsqueeze(0)

    with torch.no_grad():
        log_probs_out, values_out = model(model_in)
        prior_probs_out = F.softmax(log_probs_out, dim=1)

    values_np = values_out.detach().cpu().numpy().squeeze()
    prior_probs_np = prior_probs_out.detach().cpu().numpy().squeeze()

    if show_hints:
        print(f"Value: {values_np}")
        print(f"Prior Probabilities: {prior_probs_np}")
    return values_np, prior_probs_np


def play_against_model(num_mcts_searches=30, show_hints=True):
    # model_path = "model_checkpoints/best/testrun1_04082021_151335_best_7.tar"
    model_path = "model_checkpoints/best/train_steps_1000_08082021_062521_best_91.tar"
    model_id = basename(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    n_rows, n_cols, n_to_win = 6, 7, 4
    game = ConnectNGame(n_rows, n_cols, n_to_win)

    input_shape = (2, game.n_rows, game.n_cols)
    num_filters = 64
    num_residual_blocks = 3
    val_hidden_size = 20

    model = CNNModel(input_shape, num_filters, num_residual_blocks, val_hidden_size, game.n_cols).to(device)
    load_checkpoint(model_path, model, device=device)
    model.eval()

    m = MonteCarloTreeSearch(model, game, device=device)

    print(f"Playing against {model_id}")
    print(f"Pick a color, {ConnectNGame.val_to_char(Player.FIRST_PLAYER.value)}={Player.FIRST_PLAYER.value}, "
          f"{ConnectNGame.val_to_char(Player.SECOND_PLAYER.value)}={Player.SECOND_PLAYER.value}")

    human_col = int(input())
    if human_col == Player.FIRST_PLAYER.value:
        human_player = Player.FIRST_PLAYER
    elif human_col == Player.SECOND_PLAYER.value:
        human_player = Player.SECOND_PLAYER
    else:
        raise ValueError("Invalid Input given")

    model_player = switch_player(human_player)

    while True:
        whose_turn = np.random.choice([human_player, model_player])
        state = game.initial_state()

        while True:
            print("=" * 50)
            print("State")
            game.render(state)
            predict_with_model(game, model, state, device, show_hints=show_hints)

            m.traverse_and_backup(num_mcts_searches, state, human_player)
            high_temp_probs = m.policy_value(state, 1)
            low_temp_probs = m.policy_value(state, 0)
            if show_hints:
                print(f"Probabilities: {high_temp_probs}")
                if whose_turn == human_player:
                    print(f"Suggested Action: {low_temp_probs.argmax()}")

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

                if won:
                    print("You won, congratulations")
                    break
                elif len(game.valid_actions(state)) == 0:
                    # draw
                    print("Draw!")
                    break

            else:
                print("My Turn")

                probs = m.policy_value(state, 0)
                model_action = np.random.choice(m.num_actions, p=probs)
                state, won = game.move(state, model_action, model_player)

                print(f"I chose action {model_action}")

                if won:
                    print("I won")
                    break
                elif len(game.valid_actions(state)) == 0:
                    # draw
                    print("Draw!")
                    break

            whose_turn = human_player if whose_turn == model_player else model_player

        print("New Game")

    pass


def play_against_model_without_mcts():
    # model_path = "model_checkpoints/best_old/testrun1_26072021_100008_best_93.tar"
    model_path = "model_checkpoints/best/testrun1_05082021_054909_best_53.tar"
    model_id = basename(model_path)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

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

    human_col = int(input())
    if human_col == Player.FIRST_PLAYER.value:
        human_player = Player.FIRST_PLAYER
    elif human_col == Player.SECOND_PLAYER.value:
        human_player = Player.SECOND_PLAYER
    else:
        raise ValueError("Invalid Input given")

    model_player = switch_player(human_player)

    while True:
        whose_turn = np.random.choice([human_player, model_player])
        state = game.initial_state()

        while True:
            print("=" * 50)
            print("State")
            game.render(state)
            values_np, prior_probs_np = predict_with_model(game, model, state, device)

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

                if won:
                    print("You won, congratulations")
                    break
                elif len(game.valid_actions(state)) == 0:
                    # draw
                    print("Draw!")
                    break

            else:
                # Play without MCTS here
                print("My Turn")

                model_action = prior_probs_np.argmax()
                state, won = game.move(state, model_action, model_player)

                print(f"I chose action {model_action}")

                if won:
                    print("I won")
                    break
                elif len(game.valid_actions(state)) == 0:
                    # draw
                    print("Draw!")
                    break

            whose_turn = human_player if whose_turn == model_player else model_player

        print("New Game")

    pass


if __name__ == "__main__":
    play_against_model(num_mcts_searches=30, show_hints=True)
