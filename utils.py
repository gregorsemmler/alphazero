import logging
import signal

import torch


logger = logging.getLogger(__name__)


CHECKPOINT_MODEL = "model"
CHECKPOINT_OPTIMIZER = "optimizer"
CHECKPOINT_MODEL_ID = "model_id"


def load_checkpoint(path, model, optimizer=None, device="cpu"):
    state = torch.load(path, map_location=device)
    model.load_state_dict(state[CHECKPOINT_MODEL])
    if optimizer is not None:
        optimizer.load_state_dict(state[CHECKPOINT_OPTIMIZER])
    epoch = state.get(CHECKPOINT_MODEL_ID)
    return epoch


def save_checkpoint(path, model, optimizer=None, model_id=None):
    torch.save({
        CHECKPOINT_MODEL: model.state_dict(),
        CHECKPOINT_OPTIMIZER: optimizer.state_dict() if optimizer is not None else None,
        CHECKPOINT_MODEL_ID: model_id,
    }, path)


class GracefulExit(object):

    def __init__(self):
        self.run = True
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, *args):
        logger.info("Termination Signal received. Exiting gracefully")
        self.run = False
