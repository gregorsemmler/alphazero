import logging
import pickle
from collections import deque
from copy import deepcopy
from datetime import datetime
from os import makedirs
from os.path import join
from timeit import default_timer as timer

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter

from game import ConnectNGame
from mcts import MonteCarloTreeSearch, GameHistoryEntry
from model import CNNModel
from utils import save_checkpoint, load_checkpoint, GracefulExit

logger = logging.getLogger(__name__)


def evaluate(game, model1, model2, num_mcts_searches, num_matches, num_input_states, device=torch.device("cpu")):

    mcts1 = MonteCarloTreeSearch(model1, game, num_input_states=num_input_states, device=device)
    mcts2 = MonteCarloTreeSearch(model2, game, num_input_states=num_input_states, device=device)

    n_wins1 = 0

    for match_idx in range(num_matches):
        s = timer()
        r, m_s = mcts1.play_match(num_mcts_searches, lambda x: 0.0, other_mcts=mcts2)
        e = timer()
        m_t = e - s

        if r > 0:
            n_wins1 += 1

        logger.info(f"Eval Match {match_idx} with {m_s} steps took {m_t:.3f} seconds ({m_s / m_t:.3f} steps/s) "
                    f"({n_wins1}/{match_idx+1} wins)")

    return n_wins1 / num_matches


def scheduler_step(scheduler, writer=None, log_idx=None, log_prefix=None):
    try:
        scheduler.step()
    except UnboundLocalError as e:  # For catching OneCycleLR errors when stepping too often
        return

    if all([el is not None for el in [writer, log_idx, log_prefix]]):
        current_lr = scheduler.optimizer.param_groups[0]["lr"]
        writer.add_scalar(f"{log_prefix}/lr", current_lr, log_idx)


def get_state_list(rp_buffer, rp_idx, count):
    if rp_idx < 0 or rp_idx >= len(rp_buffer):
        raise ValueError(f"Index for replay buffer is out of range: {rp_idx}")

    game_id = rp_buffer[rp_idx].game_id
    new_idx = rp_idx
    while True:
        if new_idx <= 0:
            break
        if rp_idx - new_idx >= count - 1:
            break

        cur_game_id = rp_buffer[new_idx - 1].game_id

        if cur_game_id != game_id:
            break
        new_idx -= 1
    return [rp_buffer[ix] for ix in range(new_idx, rp_idx + 1)]


def main():
    logging.basicConfig(level=logging.INFO)

    checkpoint_path = "model_checkpoints"
    best_models_path = join(checkpoint_path, "best")
    run_id = f"two_states_in_{datetime.now():%d%m%Y_%H%M%S}"
    model_id = f"{run_id}"
    writer = SummaryWriter(comment=f"-{run_id}")

    makedirs(checkpoint_path, exist_ok=True)
    makedirs(best_models_path, exist_ok=True)

    device_token = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_token)

    n_rows, n_cols, n_to_win = 6, 7, 4
    game = ConnectNGame(n_rows, n_cols, n_to_win)

    num_input_states = 2
    input_shape = (2 * num_input_states, game.n_rows, game.n_cols)
    num_filters = 64
    num_residual_blocks = 3
    val_hidden_size = 20
    model = CNNModel(input_shape, num_filters, num_residual_blocks, val_hidden_size, game.n_cols).to(device)

    # replay_buffer_path = None
    # pretrained_model_path = None
    replay_buffer_path = "replay_buffers/rb__two_states_in_22082021_074641__500000"
    pretrained_model_path = "model_checkpoints/best/two_states_in_22082021_074641_best_168.tar"
    pretrained = pretrained_model_path is not None

    replay_buffer_size = 500000

    if isinstance(replay_buffer_path, str):
        with open(replay_buffer_path, "rb") as f:
            replay_buffer = pickle.load(f)
        replay_buffer = deque(replay_buffer, maxlen=replay_buffer_size)
        logger.info(f"Loaded replay buffer from \"{replay_buffer_path}\".")
    else:
        replay_buffer = deque(maxlen=replay_buffer_size)

    if pretrained:
        load_checkpoint(pretrained_model_path, model, device=device)
        logger.info(f"Loaded pretrained model from \"{pretrained_model_path}\".")

    best_model = deepcopy(model)
    mcts = MonteCarloTreeSearch(best_model, game, num_input_states=num_input_states, device=device)

    best_win_ratio = 0.55
    batch_size = 256
    lr = 0.1
    momentum = 0.9
    l2_regularization = 1e-4
    train_steps = 100
    min_size_to_train = 5000
    save_all_eval_checkpoints = False

    def simple_tau_sched(x):
        return 0 if x > 30 else 1

    num_mcts_searches = 10
    num_games_played = 50
    milestones = [int(el) for el in [200e3, 400e3, 600e3]]  # Milestones for mini-batch lr scheduling steps from paper

    num_eval_mcts_searches = 10
    num_eval_games = 30

    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=l2_regularization)
    scheduler = MultiStepLR(optimizer, milestones=milestones)

    curr_epoch_idx = 0
    curr_train_batch_idx = 0
    best_model_idx = 0
    n_total_train_games = 0
    n_total_eval_games = 0

    graceful_exit = GracefulExit()
    save_rp_buffer_on_exit = True
    replay_buffers_save_path = "replay_buffers"

    while graceful_exit.run:

        for game_idx in range(num_games_played):
            s = timer()
            _, m_steps = mcts.play_match(num_mcts_searches, simple_tau_sched, replay_buffer=replay_buffer)
            e = timer()
            m_t = e - s
            logger.info(f"Epoch {curr_epoch_idx}: Match {game_idx} "
                        f"with {m_steps} steps took {m_t:.3f} seconds ({m_steps / m_t:.3f} steps/s). "
                        f"Replay Buffer Size: {len(replay_buffer)}")

        n_total_train_games += num_games_played
        writer.add_scalar("epoch/games_played", n_total_train_games, curr_epoch_idx)
        writer.add_scalar("epoch/replay_buffer_size", len(replay_buffer), curr_epoch_idx)

        if len(replay_buffer) < min_size_to_train:
            logger.info(f"Epoch {curr_epoch_idx}: Minimum replay buffer size "
                        f"for training not yet reached {len(replay_buffer)}/{min_size_to_train}")
            continue

        epoch_loss = 0.0
        epoch_policy_loss = 0.0
        epoch_value_loss = 0.0
        count_batches = 0

        model.train()
        for _ in range(train_steps):
            valid_start_idx = 0

            while True:
                cur_entry: GameHistoryEntry = replay_buffer[valid_start_idx]
                if cur_entry.state_idx == 0:
                    break
                if valid_start_idx >= num_input_states - 1:
                    break
                valid_start_idx += 1

            batch_indices = np.random.choice(range(valid_start_idx, len(replay_buffer)), size=batch_size, replace=False)
            batch_samples = [get_state_list(replay_buffer, b_idx, num_input_states) for b_idx in batch_indices]
            states_t = torch.stack(
                [mcts.game.states_to_tensor([e.state for e in lst], num_input_states, device=device) for lst in
                 batch_samples])

            vals_t = torch.tensor([lst[-1].value for lst in batch_samples], device=device, dtype=torch.float32)
            probs_t = torch.tensor([lst[-1].probs for lst in batch_samples], device=device, dtype=torch.float32)

            log_probs_out, val_out = model(states_t)

            value_loss = F.mse_loss(val_out.squeeze(), vals_t)
            policy_loss = -F.log_softmax(log_probs_out, dim=1) * probs_t
            policy_loss = policy_loss.sum(dim=1).mean()
            loss = value_loss + policy_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            scheduler_step(scheduler, writer, curr_train_batch_idx, "batch")  # Batchwise scheduler in this case

            batch_loss = loss.item()
            writer.add_scalar("batch/loss", batch_loss, curr_train_batch_idx)
            writer.add_scalar("batch/policy_loss", policy_loss.item(), curr_train_batch_idx)
            writer.add_scalar("batch/value_loss", value_loss.item(), curr_train_batch_idx)

            best_model_log = f"Best Model Idx: {best_model_idx - 1} " if best_model_idx > 0 else ""
            logger.info(f"Epoch {curr_epoch_idx}: Training - "
                        f"{best_model_log}Batch: {curr_train_batch_idx}: Loss {batch_loss}, "
                        f"(Policy Loss: {policy_loss.item()}, Value Loss: {value_loss.item()})")

            curr_train_batch_idx += 1
            count_batches += 1

            epoch_loss += batch_loss
            epoch_policy_loss += policy_loss.item()
            epoch_value_loss += value_loss.item()

        epoch_loss /= max(1.0, count_batches)
        epoch_policy_loss /= max(1.0, count_batches)
        epoch_value_loss /= max(1.0, count_batches)

        if save_all_eval_checkpoints:
            save_fname = f"{model_id}_{curr_epoch_idx}.tar"
            save_checkpoint(join(checkpoint_path, save_fname), model, model_id=curr_epoch_idx)
            logging.info(f"Saved checkpoint {curr_epoch_idx} after {count_batches} steps")

        writer.add_scalar("epoch/loss", epoch_loss, curr_epoch_idx)
        writer.add_scalar("epoch/policy_loss", epoch_policy_loss, curr_epoch_idx)
        writer.add_scalar("epoch/value_loss", epoch_value_loss, curr_epoch_idx)
        logger.info(f"Epoch {curr_epoch_idx}: Loss: {epoch_loss}, "
                    f"Policy Loss: {epoch_policy_loss}, Value Loss: {epoch_value_loss}")

        curr_epoch_idx += 1

        win_ratio = evaluate(game, model, best_model, num_eval_mcts_searches, num_eval_games, num_input_states,
                             device=device)

        n_total_eval_games += num_eval_games
        writer.add_scalar("epoch/eval_games", n_total_eval_games, curr_epoch_idx)

        logger.info(f"Evaluation against best model, win ratio: {win_ratio}")
        writer.add_scalar("epoch/win_ratio", win_ratio, curr_epoch_idx)
        if win_ratio > best_win_ratio:
            best_model.load_state_dict(model.state_dict())
            best_fname = f"{model_id}_best_{best_model_idx}.tar"
            save_checkpoint(join(best_models_path, best_fname), model, model_id=curr_epoch_idx)
            logger.info(f"Epoch {curr_epoch_idx}: New Best Model {best_model_idx} saved.")
            mcts.reset()
            best_model_idx += 1

        pass
    pass

    if save_rp_buffer_on_exit:
        makedirs(replay_buffers_save_path, exist_ok=True)
        rp_save_path = join(replay_buffers_save_path, f"rb__{model_id}__{len(replay_buffer)}")
        with open(rp_save_path, "wb+") as f:
            pickle.dump(replay_buffer, f)

        logger.info(f"Saved current replay buffer to \"{rp_save_path}\".")


if __name__ == "__main__":
    main()
