import logging
from collections import deque
from copy import deepcopy
from datetime import datetime
from os import makedirs
from os.path import join

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter

from game import ConnectNGame
from mcts import MCTS
from model import CNNModel
from utils import save_checkpoint, load_checkpoint

logger = logging.getLogger(__name__)


def evaluate(game, model1, model2, num_mcts_searches, num_matches, device=torch.device("cpu")):
    mcts1 = MCTS(model1, game, search_batch_size=num_mcts_searches, device=device)
    mcts2 = MCTS(model2, game, search_batch_size=num_mcts_searches, device=device)

    n_wins1, n_wins2 = 0, 0

    for _ in range(num_matches):
        r, _ = mcts1.play_match(num_mcts_searches, lambda x: 0.0, mcts2)

        if r > 0:
            n_wins1 += 1

    return n_wins1 / num_matches


def scheduler_step(scheduler, writer=None, log_idx=None, log_prefix=None):
    try:
        scheduler.step()
    except UnboundLocalError as e:  # For catching OneCycleLR errors when stepping too often
        return

    if all([el is not None for el in [writer, log_idx, log_prefix]]):
        current_lr = scheduler.optimizer.param_groups[0]["lr"]
        writer.add_scalar(f"{log_prefix}/lr", current_lr, log_idx)


def main():
    logging.basicConfig(level=logging.INFO)

    checkpoint_path = "model_checkpoints"
    best_models_path = join(checkpoint_path, "best")
    pretrained_model_path = None
    pretrained = pretrained_model_path is not None

    run_id = "testrun1"
    model_id = f"{run_id}"
    writer = SummaryWriter(comment=f"-{model_id}-{run_id}")

    makedirs(checkpoint_path, exist_ok=True)
    makedirs(best_models_path, exist_ok=True)

    device_token = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_token)

    n_rows, n_cols, n_to_win = 6, 7, 4
    game = ConnectNGame(n_rows, n_cols, n_to_win)

    input_shape = (2, game.n_rows, game.n_cols)
    num_filters = 64
    num_residual_blocks = 5
    val_hidden_size = 20
    model = CNNModel(input_shape, num_filters, num_residual_blocks, val_hidden_size, game.n_cols).to(device)

    if pretrained:
        load_checkpoint(pretrained_model_path, model, device=device)
        logger.info(f"Loaded pretrained model from \"{pretrained_model_path}\"")

    best_model = deepcopy(model)
    mcts = MCTS(best_model, game, device=device)

    best_win_ratio = 0.55
    replay_buffer_size = 100000
    batch_size = 256
    lr = 0.1
    momentum = 0.9
    l2_regularization = 1e-4
    train_steps = 20
    min_size_to_train = 5000
    save_all_eval_checkpoints = True

    def simple_tau_sched(x):
        return 0 if x > 30 else 1

    num_mcts_searches = 100
    num_games_played = 50
    milestones = [int(el) for el in [200e3, 400e3, 600e3]]  # Milestones for mini-batch lr scheduling steps from paper

    num_eval_mcts_searches = 100
    num_eval_games = 50

    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=l2_regularization)
    scheduler = MultiStepLR(optimizer, milestones=milestones)
    replay_buffer = deque(maxlen=replay_buffer_size)

    curr_epoch_idx = 0
    curr_train_batch_idx = 0
    best_model_idx = 0

    while True:

        for game_idx in range(num_games_played):
            _, match_steps = mcts.play_match(num_mcts_searches, simple_tau_sched, replay_buffer=replay_buffer)

        if len(replay_buffer) < min_size_to_train:
            continue

        epoch_loss = 0.0
        count_batches = 0

        model.train()
        for _ in range(train_steps):
            batch_samples = np.random.choice(replay_buffer, size=batch_size, replace=False)
            batch_states, batch_probs, batch_vals = zip(*batch_samples)  # TODO separate class for replay buffer?
            states_t = torch.stack([mcts.game.state_to_tensor(el, device=device) for el in batch_states])
            vals_t = torch.tensor(batch_vals, device=device)
            probs_t = torch.tensor(batch_probs, device=device)

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
            writer.add_scalar("train_batch/loss", batch_loss, curr_train_batch_idx)
            logger.info(f"Training - Model Idx: {curr_epoch_idx} Batch: {curr_train_batch_idx}: Loss {batch_loss}")
            curr_train_batch_idx += 1
            count_batches += 1

            epoch_loss += batch_loss

        epoch_loss /= max(1.0, count_batches)

        if save_all_eval_checkpoints:
            save_fname = f"{model_id}_{curr_epoch_idx}_{datetime.now():%d%m%Y_%H%M%S}.tar"
            save_checkpoint(join(checkpoint_path, save_fname), model, optimizer, model_id=curr_epoch_idx)
            logging.info(f"Saved {curr_epoch_idx} checkpoint after {curr_train_batch_idx} steps")

        writer.add_scalar("train_epoch/loss", epoch_loss, curr_epoch_idx)
        logger.info(f"Training - Epoch {curr_epoch_idx}: Loss {epoch_loss}")

        curr_epoch_idx += 1

        win_ratio = evaluate(game, model, best_model, num_eval_mcts_searches, num_eval_games, device=device)
        logger.info(f"Evaluation against best model, win ratio: {win_ratio}")
        if win_ratio > best_win_ratio:
            best_model.load_state_dict(model.state_dict())
            best_fname = f"{model_id}_best_{best_model_idx}.tar"
            save_checkpoint(join(best_models_path, best_fname), model, optimizer, model_id=curr_epoch_idx)
            logger.info(f"New Best Model {best_model_idx} saved")
            mcts.reset()

        pass
    pass


if __name__ == "__main__":
    main()
