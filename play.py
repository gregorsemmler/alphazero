from collections import deque
from os.path import basename

import numpy as np
import torch
import torch.nn.functional as F

from game import ConnectNGame, Player, switch_player
from mcts import MonteCarloTreeSearch
from model import CNNModel
from utils import load_checkpoint
from visualize import visualize_connect_n_game


def predict_with_model(game, model, states, num_input_states, device, show_hints=True):
    model_in = game.states_to_tensor(states, num_input_states, device=device).unsqueeze(0)

    with torch.no_grad():
        log_probs_out, values_out = model(model_in)
        prior_probs_out = F.softmax(log_probs_out, dim=1)

    values_np = values_out.detach().cpu().numpy().squeeze()
    prior_probs_np = prior_probs_out.detach().cpu().numpy().squeeze()

    if show_hints:
        print(f"Value: {values_np}")
        print(f"Prior Probabilities: {prior_probs_np}")
    return values_np, prior_probs_np


def play_against_model(num_mcts_searches=30, show_hints=True, viz_with_image=True):
    model_path = "model_checkpoints/best/two_states_in_21082021_053954_best_104.tar"
    model_id = basename(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_rows, n_cols, n_to_win = 6, 7, 4
    game = ConnectNGame(n_rows, n_cols, n_to_win)

    num_input_states = 2
    input_shape = (2 * num_input_states, game.n_rows, game.n_cols)
    num_filters = 64
    num_residual_blocks = 3
    val_hidden_size = 20

    model = CNNModel(input_shape, num_filters, num_residual_blocks, val_hidden_size, game.n_cols).to(device)
    load_checkpoint(model_path, model, device=device)
    model.eval()

    m = MonteCarloTreeSearch(model, game, num_input_states, device=device)

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
        last_states = deque([state], maxlen=num_input_states)

        while True:
            print("=" * 50)
            print("State")
            predict_with_model(game, model, last_states, num_input_states, device, show_hints=show_hints)

            m.traverse_and_backup(num_mcts_searches, state, human_player)
            high_temp_probs = m.policy_value(state, 1)
            low_temp_probs = m.policy_value(state, 0)

            if show_hints:
                print(f"Probabilities: {high_temp_probs}")
                if whose_turn == human_player:
                    print(f"Suggested Action: {low_temp_probs.argmax()}")

            game.render(state)
            if viz_with_image:
                show_probs = high_temp_probs if show_hints else np.zeros_like(high_temp_probs)
                visualize_connect_n_game(state, show_probs)

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
                last_states.append(state)

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
                last_states.append(state)

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


def play_against_model_without_mcts(viz_with_image=True, show_hints=True):
    model_path = "model_checkpoints/best/two_states_in_21082021_053954_best_102.tar"
    model_id = basename(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    n_rows, n_cols, n_to_win = 6, 7, 4
    game = ConnectNGame(n_rows, n_cols, n_to_win)

    num_input_states = 2
    input_shape = (2 * num_input_states, game.n_rows, game.n_cols)
    num_filters = 64
    num_residual_blocks = 3
    val_hidden_size = 20

    model = CNNModel(input_shape, num_filters, num_residual_blocks, val_hidden_size, game.n_cols).to(device)
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
        last_states = deque([state], maxlen=num_input_states)

        while True:
            print("=" * 50)
            print("State")
            values_np, prior_probs_np = predict_with_model(game, model, last_states, num_input_states, device)

            game.render(state)

            if viz_with_image:
                show_probs = prior_probs_np if show_hints else np.zeros_like(prior_probs_np)
                visualize_connect_n_game(state, show_probs)

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
                last_states.append(state)

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
                last_states.append(state)

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
    # play_against_model(num_mcts_searches=30, show_hints=True)
    play_against_model_without_mcts()
