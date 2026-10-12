"""Configuration, reproducibility settings, dataset manifests, and output protection."""
from pathlib import Path
from datetime import datetime, timezone
import csv
import json
import random

import numpy as np
import torch
import yaml
from torchvision import datasets

from .data import _stratified_split

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_RUN = ROOT / "artifacts/kaggle_run"


def load_config(path=ROOT / "configs/default.yaml"):
    return yaml.safe_load(Path(path).read_text())


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(name="auto"):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable. Use --device cpu or auto.")
    return torch.device(name)


def prepare_output(path=None):
    if path is None:
        path = ROOT / "artifacts/runs" / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f_UTC")
    path = Path(path)
    if path.resolve().is_relative_to(ORIGINAL_RUN.resolve()):
        raise ValueError("Original Kaggle artifacts are read-only evidence. Choose a new output folder.")
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Output folder is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def validate_dataset_root(root):
    root = Path(root)
    for split in ("train", "test"):
        directory = root / split
        if not directory.is_dir():
            raise FileNotFoundError(f"Missing {directory}. Download the casting dataset; see data/README.md.")
        folders = {p.name for p in directory.iterdir() if p.is_dir()}
        if folders != {"def_front", "ok_front"}:
            raise ValueError(f"{directory} must contain exactly def_front/ and ok_front/.")
    return root


def write_manifest(cfg, output):
    root = validate_dataset_root(cfg["data"]["root"])
    training = datasets.ImageFolder(root / "train")
    test = datasets.ImageFolder(root / "test")
    train_idx, val_idx = _stratified_split(training.targets, cfg["data"]["val_split"], cfg["data"]["seed"])
    if set(train_idx) & set(val_idx):
        raise ValueError("Training and validation indices overlap.")
    rows = []
    for split, folder, indices in [("train", training, train_idx), ("validation", training, val_idx),
                                   ("test", test, range(len(test)))]:
        for idx in indices:
            path, label = folder.samples[idx]
            rows.append({"split": split, "path": Path(path).relative_to(root).as_posix(),
                         "label": int(folder.classes[label] == "def_front")})
    with (output / "split_manifest.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "path", "label"])
        writer.writeheader(); writer.writerows(rows)
    summary = {s: {"total": sum(r["split"] == s for r in rows),
                   "ok": sum(r["split"] == s and r["label"] == 0 for r in rows),
                   "defect": sum(r["split"] == s and r["label"] == 1 for r in rows)}
               for s in ("train", "validation", "test")}
    (output / "dataset_summary.json").write_text(json.dumps(summary, indent=2))
    return rows, summary
