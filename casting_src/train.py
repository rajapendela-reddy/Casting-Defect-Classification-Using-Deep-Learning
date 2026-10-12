"""Train the Kaggle experiment locally; checkpoint selection uses validation defect F1."""
import argparse
import copy
import gc
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import yaml

from .checkpoints import load_checkpoint
from .common import ROOT, load_config, prepare_output, resolve_device, set_seed, validate_dataset_root, write_manifest
from .data import build_dataloaders
from .evaluate import collect_probs, evaluate_selected
from .gradcam import denormalize
from .metrics import binary_metrics
from .model import build_model
from .predict import predict_image


def train_architecture(arch, cfg, output, device):
    set_seed(cfg["data"]["seed"])
    training, validation, _, weights = build_dataloaders(cfg)
    run_cfg = copy.deepcopy(cfg); run_cfg["model"]["arch"] = arch
    model = build_model(run_cfg).to(device)
    if not cfg["train"]["use_class_weights"]:
        raise ValueError("This experiment uses class-weighted loss; keep use_class_weights=true.")
    criterion = torch.nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    epochs, patience = cfg["train"]["epochs"], cfg["train"]["early_stopping_patience"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_score, best_epoch, history = -1.0, 0, []
    checkpoint = output / f"best_{arch}.pt"
    for epoch in range(1, epochs + 1):
        start = time.time(); model.train(); running = 0.0; n = 0
        for x, y in training:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward(); optimizer.step()
            running += loss.item() * x.size(0); n += x.size(0)
        probability, labels = collect_probs(model, validation, device)
        metrics = binary_metrics(labels, probability, 0.5)
        scheduler.step()
        record = {"epoch": epoch, "train_loss": running / n, **metrics}; history.append(record)
        print(f"{arch} epoch {epoch:02d}: loss={record['train_loss']:.4f} "
              f"val_F1={metrics['f1_defect']:.4f} val_recall={metrics['recall_defect']:.4f} "
              f"({time.time() - start:.1f}s)", flush=True)
        if metrics["f1_defect"] > best_score:
            best_score, best_epoch = metrics["f1_defect"], epoch
            torch.save({"state_dict": model.state_dict(), "arch": arch, "config": run_cfg,
                        "best_epoch": epoch}, checkpoint)
        if epoch - best_epoch >= patience:
            print("Early stopping."); break
    del model, optimizer, scheduler, criterion
    gc.collect()
    if device.type == "cuda": torch.cuda.empty_cache()
    best_model, _ = load_checkpoint(checkpoint, device)
    probability, labels = collect_probs(best_model, validation, device)
    result = {"arch": arch, "best_epoch": best_epoch, "history": history,
              "validation": binary_metrics(labels, probability, 0.5), "checkpoint": str(checkpoint)}
    (output / f"training_{arch}.json").write_text(json.dumps(result, indent=2))
    np.savez_compressed(output / f"validation_predictions_{arch}.npz", y_true=labels, p_defect=probability)
    del best_model; gc.collect()
    if device.type == "cuda": torch.cuda.empty_cache()
    return result


def run_experiment(cfg, output=None, device_name="auto"):
    cfg = copy.deepcopy(cfg)
    cfg["data"]["root"] = str(validate_dataset_root(cfg["data"]["root"]))
    if cfg["train"]["monitor"] != "f1_defect" or cfg["eval"]["threshold"] != 0.5:
        raise ValueError("The completed experiment selects on f1_defect at threshold 0.5.")
    if cfg["model"]["freeze_backbone"]:
        raise ValueError("The completed experiment fine-tunes the full backbone; freeze_backbone must be false.")
    if cfg["train"]["epochs"] < 1 or cfg["train"]["early_stopping_patience"] < 1:
        raise ValueError("Epochs and patience must be positive.")
    if cfg["data"]["image_size"] < 32 or cfg["data"]["batch_size"] < 1 or cfg["data"]["num_workers"] < 0:
        raise ValueError("Image size must be >=32, batch size positive, workers nonnegative.")
    if not 0 < cfg["data"]["val_split"] < 1:
        raise ValueError("Validation fraction must lie strictly between 0 and 1.")
    architectures = cfg["experiment"]["architectures"]
    if not architectures or len(set(architectures)) != len(architectures):
        raise ValueError("Supply at least one architecture without duplicates.")
    if not set(architectures) <= {"simple_cnn", "resnet18", "efficientnet_b0"}:
        raise ValueError("Unknown architecture.")
    if cfg["experiment"]["bootstrap_samples"] < 2:
        raise ValueError("At least two bootstrap samples are required.")
    device = resolve_device(device_name)
    output = prepare_output(output)
    cfg["paths"]["output_dir"] = str(output)
    (output / "config.yaml").write_text(yaml.safe_dump(cfg))
    set_seed(cfg["data"]["seed"])
    _, validation, test, _ = build_dataloaders(cfg)
    rows, summary = write_manifest(cfg, output)
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for i, ax in enumerate(axes.flat):
        x, y = validation.dataset[i % len(validation.dataset)]
        ax.imshow(denormalize(x)); ax.set_title(["ok", "defect"][y]); ax.axis("off")
    fig.tight_layout(); fig.savefig(output / "sample_images.png", dpi=150); plt.close(fig)
    results = [train_architecture(arch, cfg, output, device) for arch in architectures]
    winner = max(results, key=lambda r: r["validation"]["f1_defect"])
    model, checkpoint = load_checkpoint(winner["checkpoint"], device)
    print("Selected using validation F1:", winner["arch"], "| best epoch:", winner["best_epoch"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for r in results:
        epochs = [h["epoch"] for h in r["history"]]
        axes[0].plot(epochs, [h["train_loss"] for h in r["history"]], label=r["arch"])
        axes[1].plot(epochs, [h["f1_defect"] for h in r["history"]], label=r["arch"])
    for ax in axes: ax.set_xlabel("Epoch"); ax.legend(); ax.grid(alpha=0.2)
    axes[0].set_ylabel("Training loss"); axes[1].set_ylabel("Validation defect F1")
    fig.tight_layout(); fig.savefig(output / "convergence.png", dpi=150); plt.close(fig)
    report = evaluate_selected(model, checkpoint, cfg, validation, test, rows, output, device,
                               cfg["experiment"]["bootstrap_samples"])
    first_test = next(r for r in rows if r["split"] == "test")
    example = predict_image(model, {**checkpoint, "operating_threshold": report["threshold_selection"]["threshold"]},
                            Path(cfg["data"]["root"]) / first_test["path"], device)
    (output / "example_prediction.json").write_text(json.dumps(example, indent=2))
    metadata = {"python": platform.python_version(), "torch": str(torch.__version__),
                "torchvision": str(torchvision.__version__), "numpy": np.__version__, "device": str(device),
                "seed": cfg["data"]["seed"], "dataset_root": cfg["data"]["root"], "architectures": architectures,
                "pretrained_backbones": cfg["model"]["pretrained"], "selected_arch": winner["arch"],
                "operating_threshold": report["threshold_selection"]["threshold"], "dataset_summary": summary,
                "bootstrap_samples": cfg["experiment"]["bootstrap_samples"]}
    if device.type == "cuda": metadata["gpu"] = torch.cuda.get_device_name(0)
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    (output / "environment.txt").write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True))
    source = output / "source"; source.mkdir()
    for path in (ROOT / "casting_src").glob("*.py"): shutil.copy2(path, source / path.name)
    print("Saved run:", output.resolve())
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/default.yaml")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--architectures", nargs="+", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args(); cfg = load_config(args.config)
    cfg["data"]["root"] = str(args.data_root or ROOT / cfg["data"]["root"])
    for key, value in [("image_size", args.image_size), ("batch_size", args.batch_size), ("num_workers", args.workers)]:
        if value is not None: cfg["data"][key] = value
    if args.epochs is not None: cfg["train"]["epochs"] = args.epochs
    if args.architectures: cfg["experiment"]["architectures"] = args.architectures
    if args.no_pretrained: cfg["model"]["pretrained"] = False
    run_experiment(cfg, args.output, args.device)


if __name__ == "__main__":
    main()
