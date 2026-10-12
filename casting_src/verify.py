"""Verify stored predictions, metrics, split assignments, threshold choice, and optional checkpoints."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

from .metrics import binary_metrics
from .threshold import select_threshold

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_RUN = ROOT / "artifacts/kaggle_run"


def assert_metrics(actual, expected):
    for key, value in actual.items():
        if key not in expected: raise ValueError(f"Missing metric: {key}")
        np.testing.assert_allclose(value, expected[key], atol=1e-12, rtol=0, err_msg=f"Metric mismatch: {key}")


def read_probabilities(path):
    with np.load(path, allow_pickle=False) as data:
        y, p = data["y_true"].copy(), data["p_defect"].copy()
    if y.ndim != 1 or p.shape != y.shape or not len(y):
        raise ValueError("Invalid prediction arrays.")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Labels or probabilities are invalid.")
    return y, p


def verify_results(run=ORIGINAL_RUN, check_checkpoints=False):
    run = Path(run)
    metadata = json.loads((run / "run_metadata.json").read_text())
    cfg = yaml.safe_load((run / "config.yaml").read_text())
    summary = json.loads((run / "dataset_summary.json").read_text())
    manifest = list(csv.DictReader((run / "split_manifest.csv").open()))
    paths = [r["path"] for r in manifest]
    if len(set(paths)) != len(paths): raise ValueError("Repeated paths in dataset partitions.")
    for r in manifest:
        if r["split"] not in {"train", "validation", "test"}: raise ValueError("Invalid split label.")
        parts = Path(r["path"]).parts
        if len(parts) != 3 or ".." in parts or parts[1] not in {"def_front", "ok_front"}:
            raise ValueError("Unexpected dataset path.")
        if parts[0] != ("test" if r["split"] == "test" else "train"):
            raise ValueError("Test-folder membership is inconsistent.")
        if int(r["label"]) != int(parts[1] == "def_front"): raise ValueError("Incorrect class convention.")
    for split in ("train", "validation", "test"):
        labels = [int(r["label"]) for r in manifest if r["split"] == split]
        counts = Counter(labels)
        actual = {"total": len(labels), "ok": counts[0], "defect": counts[1]}
        if actual != summary[split] or actual != metadata["dataset_summary"][split]:
            raise ValueError("Dataset counts disagree.")
    # Recreate NumPy's exact seeded class-wise split from ImageFolder's sorted path order.
    training_rows = sorted([r for r in manifest if r["split"] != "test"], key=lambda r: r["path"])
    original_targets = np.array([0 if int(r["label"]) else 1 for r in training_rows])
    rng = np.random.default_rng(cfg["data"]["seed"])
    ti, vi = [], []
    for cls in np.unique(original_targets):
        indices = np.flatnonzero(original_targets == cls); rng.shuffle(indices)
        n_val = int(round(len(indices) * cfg["data"]["val_split"]))
        vi.extend(indices[:n_val].tolist()); ti.extend(indices[n_val:].tolist())
    rng.shuffle(ti); rng.shuffle(vi)
    for split, indices in [("train", ti), ("validation", vi)]:
        actual_paths = [r["path"] for r in manifest if r["split"] == split]
        if actual_paths != [training_rows[i]["path"] for i in indices]:
            raise ValueError("Split manifest differs from the seeded stratified split.")
    test_rows = [r for r in manifest if r["split"] == "test"]
    if [r["path"] for r in test_rows] != sorted(r["path"] for r in test_rows):
        raise ValueError("Test manifest is not in ImageFolder order.")
    validation_labels = np.array([int(r["label"]) for r in manifest if r["split"] == "validation"])
    results = []
    for arch in metadata["architectures"]:
        result = json.loads((run / f"training_{arch}.json").read_text())
        y, p = read_probabilities(run / f"validation_predictions_{arch}.npz")
        np.testing.assert_array_equal(y, validation_labels)
        assert_metrics(binary_metrics(y, p, 0.5), result["validation"])
        best = max(result["history"], key=lambda h: h["f1_defect"])
        if best["epoch"] != result["best_epoch"]: raise ValueError("Best epoch is inconsistent.")
        for key in result["validation"]:
            np.testing.assert_allclose(best[key], result["validation"][key], atol=1e-12, rtol=0)
        results.append(result)
    winner = max(results, key=lambda r: r["validation"]["f1_defect"])
    if winner["arch"] != metadata["selected_arch"]: raise ValueError("Validation-F1 selection is inconsistent.")
    report = json.loads((run / "evaluation_report.json").read_text())
    if winner["arch"] != report["selected_arch"] or winner["best_epoch"] != report["best_epoch"]:
        raise ValueError("Final report does not describe the selected checkpoint.")
    selected_y, selected_p = read_probabilities(run / f"validation_predictions_{winner['arch']}.npz")
    costs = report["cost_assumptions"]
    info = select_threshold(selected_y, selected_p, cost_fn=costs["false_negative"], cost_fp=costs["false_positive"],
                            n_boot=metadata.get("bootstrap_samples", 200), seed=metadata["seed"])
    saved_info = json.loads((run / "threshold_selection.json").read_text())
    for key in info:
        for saved in [saved_info, report["threshold_selection"]]:
            np.testing.assert_allclose(info[key], saved[key], atol=1e-12, rtol=0)
    np.testing.assert_allclose(info["threshold"], metadata["operating_threshold"], atol=1e-12, rtol=0)
    y, p = read_probabilities(run / "test_predictions.npz")
    np.testing.assert_array_equal(y, [int(r["label"]) for r in test_rows])
    assert_metrics(binary_metrics(y, p, 0.5), report["test_default"])
    assert_metrics(binary_metrics(y, p, info["threshold"]), report["test_operating"])
    prediction_rows = list(csv.DictReader((run / "test_predictions.csv").open()))
    if len(prediction_rows) != len(test_rows): raise ValueError("Test CSV length mismatch.")
    for row, item, label, probability in zip(prediction_rows, test_rows, y, p):
        if row["path"] != item["path"] or int(row["actual_label"]) != label: raise ValueError("CSV alignment mismatch.")
        np.testing.assert_allclose(float(row["p_defect"]), probability, atol=1e-12, rtol=0)
        np.testing.assert_allclose(float(row["threshold"]), info["threshold"], atol=1e-12, rtol=0)
        if int(row["predicted_label"]) != int(probability >= info["threshold"]): raise ValueError("CSV classification mismatch.")
    hash_manifest = run / "artifact_manifest.json"
    if hash_manifest.exists():
        for item in json.loads(hash_manifest.read_text())["files"]:
            path = (run / item["path"]).resolve()
            if not path.is_relative_to(run.resolve()): raise ValueError("Invalid artifact path.")
            if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
                raise ValueError(f"Original artifact changed: {item['path']}")
    if check_checkpoints:
        import torch
        from .checkpoints import load_checkpoint
        selected_model, selected = load_checkpoint(run / "selected_model.pt", torch.device("cpu"))
        if selected["arch"] != winner["arch"] or selected["best_epoch"] != winner["best_epoch"]:
            raise ValueError("Selected checkpoint metadata is inconsistent.")
        np.testing.assert_allclose(selected["operating_threshold"], info["threshold"], atol=1e-12, rtol=0)
        if selected["class_names"] != ["ok", "defect"]: raise ValueError("Checkpoint class names differ.")
        for arch in metadata["architectures"]:
            model, checkpoint = load_checkpoint(run / f"best_{arch}.pt", torch.device("cpu"))
            if arch == winner["arch"]:
                for key, value in checkpoint["state_dict"].items():
                    if not torch.equal(value, selected["state_dict"][key]): raise ValueError("Selected model weights differ.")
            # This is a numerical/checkpoint plumbing probe, not dataset accuracy validation.
            with torch.no_grad():
                output = model(torch.zeros(1, 3, checkpoint["config"]["data"]["image_size"],
                                           checkpoint["config"]["data"]["image_size"]))
            if output.shape != (1, 2) or not torch.isfinite(output).all(): raise ValueError("Invalid model outputs.")
    print("Verification passed.")
    print(json.dumps({"selected_arch": winner["arch"], "best_epoch": winner["best_epoch"],
                      "dataset_summary": summary, "threshold": info["threshold"],
                      "test_default": report["test_default"], "test_operating": report["test_operating"],
                      "checkpoints_checked": check_checkpoints}, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ORIGINAL_RUN)
    parser.add_argument("--check-checkpoints", action="store_true")
    args = parser.parse_args()
    verify_results(args.run, args.check_checkpoints)


if __name__ == "__main__":
    main()
