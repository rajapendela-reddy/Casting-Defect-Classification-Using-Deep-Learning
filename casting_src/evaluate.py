"""Validation threshold selection, fixed-threshold test metrics, and Grad-CAM figures."""
import argparse
import copy
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .checkpoints import load_checkpoint
from .common import prepare_output, resolve_device, validate_dataset_root, write_manifest
from .data import CLASS_NAMES, build_dataloaders
from .gradcam import GradCAM, denormalize, overlay_heatmap
from .metrics import binary_metrics, sweep_thresholds
from .model import get_target_layer
from .threshold import select_threshold


@torch.no_grad()
def collect_probs(model, loader, device):
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        p = torch.softmax(model(x.to(device, non_blocking=True)), dim=1)[:, 1]
        probs.append(p.cpu()); labels.append(y)
    return torch.cat(probs).numpy(), torch.cat(labels).numpy()


def gradcam_grid(model, arch, dataset, indices, device, output, title, threshold):
    if not indices:
        print("No samples:", title); return
    indices = indices[:6]
    cam_engine = GradCAM(model, get_target_layer(model, arch))
    fig, axes = plt.subplots(2, len(indices), figsize=(3 * len(indices), 6), squeeze=False)
    try:
        for col, idx in enumerate(indices):
            x, y = dataset[idx]
            with torch.enable_grad():
                cam, _, p = cam_engine(x.unsqueeze(0).to(device), class_idx=1)
            image = denormalize(x)
            axes[0, col].imshow(image); axes[0, col].set_title(f"Actual: {CLASS_NAMES[y]}")
            axes[1, col].imshow(overlay_heatmap(image, cam))
            axes[1, col].set_title(f"Pred: {CLASS_NAMES[int(p >= threshold)]} | P(defect)={p:.3f}", fontsize=9)
            for row in (0, 1): axes[row, col].axis("off")
    finally:
        cam_engine.remove()
    fig.suptitle(title); fig.tight_layout(); fig.savefig(output, dpi=150); plt.close(fig)


def evaluate_selected(model, checkpoint, cfg, val_loader, test_loader, rows, output, device, n_boot=200,
                      selection_metric="validation_f1_defect"):
    seed = cfg["data"]["seed"]
    cost_fn, cost_fp = cfg["eval"]["cost_false_negative"], cfg["eval"]["cost_false_positive"]
    # Fix the operating point using validation before evaluating the test set.
    val_probs, val_y = collect_probs(model, val_loader, device)
    info = select_threshold(val_y, val_probs, cost_fn=cost_fn, cost_fp=cost_fp, n_boot=n_boot, seed=seed)
    threshold = float(info["threshold"])
    (output / "threshold_selection.json").write_text(json.dumps(info, indent=2))
    ths, costs, _ = sweep_thresholds(val_y, val_probs, cost_fn, cost_fp, n=999)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ths, costs, label="Validation cost")
    ax.axvline(0.5, ls=":", color="gray", label="Default 0.5")
    ax.axvline(threshold, ls="--", color="crimson", label=f"1-SE threshold {threshold:.3f}")
    ax.axvspan(*info["plateau"], alpha=0.15, color="green", label="Selected plateau")
    ax.set(xlabel="P(defect) threshold", ylabel=f"Cost = {cost_fn:g} × FN + {cost_fp:g} × FP")
    ax.legend(); fig.tight_layout(); fig.savefig(output / "validation_threshold_sweep.png", dpi=150); plt.close(fig)
    test_probs, test_y = collect_probs(model, test_loader, device)
    default = binary_metrics(test_y, test_probs, 0.5)
    operating = binary_metrics(test_y, test_probs, threshold)
    report = {"selected_arch": checkpoint["arch"], "selection_metric": selection_metric,
              "best_epoch": checkpoint["best_epoch"], "threshold_selection": info,
              "test_default": default, "test_operating": operating,
              "cost_assumptions": {"false_negative": cost_fn, "false_positive": cost_fp}}
    (output / "evaluation_report.json").write_text(json.dumps(report, indent=2))
    np.savez_compressed(output / "test_predictions.npz", y_true=test_y, p_defect=test_probs)
    test_rows = [r for r in rows if r["split"] == "test"]
    if len(test_rows) != len(test_y): raise ValueError("Test manifest and predictions differ in size.")
    with (output / "test_predictions.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "actual_label", "p_defect", "predicted_label", "threshold"])
        for row, y, p in zip(test_rows, test_y, test_probs):
            writer.writerow([row["path"], int(y), float(p), int(p >= threshold), threshold])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, title, m in zip(axes, ["Default threshold", "Validation-selected threshold"], [default, operating]):
        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        ax.imshow(cm, cmap="Blues")
        for (i, j), value in np.ndenumerate(cm): ax.text(j, i, str(value), ha="center", va="center")
        ax.set_xticks([0, 1], labels=CLASS_NAMES); ax.set_yticks([0, 1], labels=CLASS_NAMES)
        ax.set(xlabel="Predicted", ylabel="Actual", title=title)
    fig.tight_layout(); fig.savefig(output / "confusion_matrices.png", dpi=150); plt.close(fig)
    pred = (test_probs >= threshold).astype(int)
    correct = np.flatnonzero((test_y == 1) & (pred == 1)).tolist()
    errors = np.flatnonzero(test_y != pred).tolist()
    gradcam_grid(model, checkpoint["arch"], test_loader.dataset, correct, device,
                 output / "gradcam_correct.png", "Grad-CAM: correctly classified defects", threshold)
    gradcam_grid(model, checkpoint["arch"], test_loader.dataset, errors, device,
                 output / "gradcam_errors.png", "Grad-CAM: misclassifications", threshold)
    bundle = dict(checkpoint)
    bundle["operating_threshold"] = threshold; bundle["class_names"] = CLASS_NAMES
    torch.save(bundle, output / "selected_model.pt")
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--bootstrap-samples", type=int, default=200)
    args = parser.parse_args()
    if args.bootstrap_samples < 2: raise ValueError("At least two bootstrap samples are required.")
    device = resolve_device(args.device)
    model, ckpt = load_checkpoint(args.checkpoint, device)
    cfg = copy.deepcopy(ckpt["config"])
    cfg["data"]["root"] = str(validate_dataset_root(args.data_root))
    cfg["data"]["num_workers"] = args.workers
    output = prepare_output(args.output)
    rows, _ = write_manifest(cfg, output)
    _, validation, test, _ = build_dataloaders(cfg)
    evaluate_selected(model, ckpt, cfg, validation, test, rows, output, device, args.bootstrap_samples,
                      selection_metric="provided_checkpoint")
    print("Saved evaluation:", output.resolve())


if __name__ == "__main__":
    main()
