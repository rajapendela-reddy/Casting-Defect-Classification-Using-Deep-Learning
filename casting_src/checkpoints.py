"""Load state_dict checkpoints without re-downloading ImageNet weights."""
import copy
from pathlib import Path

import torch
from .model import build_model


def load_checkpoint(path, device=None):
    device = device or torch.device("cpu")
    checkpoint = torch.load(Path(path), map_location=device, weights_only=True)
    cfg = copy.deepcopy(checkpoint["config"])
    if checkpoint["arch"] != cfg["model"]["arch"]:
        raise ValueError("Checkpoint architecture and configuration disagree.")
    cfg["model"]["pretrained"] = False
    model = build_model(cfg).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.eval(), checkpoint
