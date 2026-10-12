"""Check original evidence, checkpoint compatibility, and synthetic-data workflow behavior."""
import copy
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from casting_src.checkpoints import load_checkpoint
from casting_src.common import ORIGINAL_RUN, load_config, prepare_output
from casting_src.data import build_dataloaders, build_transforms
from casting_src.gradcam import GradCAM
from casting_src.metrics import binary_metrics, roc_auc
from casting_src.model import build_model, get_target_layer
from casting_src.predict import predict_image
from casting_src.threshold import select_threshold
from casting_src.verify import verify_results

torch.set_num_threads(2)


def test_original_predictions_reports_and_threshold():
    report = verify_results()
    assert report["selected_arch"] == "simple_cnn"
    assert report["test_default"]["fn"] == 3 and report["test_default"]["fp"] == 0
    assert report["test_operating"]["fn"] == 0 and report["test_operating"]["fp"] == 10


def test_original_checkpoints_load_and_selected_weights_match():
    verify_results(check_checkpoints=True)


def test_auc_ties_against_pairwise_definition():
    rng = np.random.default_rng(42)
    for _ in range(5):
        y = rng.integers(0, 2, 60); p = np.round(rng.random(60), 1)
        pos, neg = p[y == 1], p[y == 0]
        expected = np.mean((pos[:, None] > neg) + 0.5 * (pos[:, None] == neg))
        assert abs(roc_auc(y, p) - expected) < 1e-12


def test_metric_confusion_convention():
    metrics = binary_metrics(np.array([0, 0, 1, 1]), np.array([.1, .8, .2, .9]), .5)
    assert [metrics[k] for k in ["tn", "fp", "fn", "tp"]] == [1, 1, 1, 1]


def test_single_point_threshold_plateau():
    info = select_threshold(np.array([0, 1]), np.array([.8, .95]),
                            grid=np.array([.1, .85, .99]), n_boot=20, seed=42)
    assert info["threshold"] == .85


@pytest.mark.parametrize("arch", ["simple_cnn", "resnet18", "efficientnet_b0"])
def test_architecture_and_gradcam(arch):
    cfg = load_config(); cfg["model"].update(arch=arch, pretrained=False)
    model = build_model(cfg).eval()
    engine = GradCAM(model, get_target_layer(model, arch))
    try:
        cam, _, probability = engine(torch.rand(1, 3, 32, 32), class_idx=1)
        assert cam.shape == (32, 32) and np.isfinite(cam).all()
        assert 0 <= cam.min() <= cam.max() <= 1 and 0 <= probability <= 1
    finally:
        engine.remove()


def test_same_image_transform_and_threshold_for_prediction(tmp_path):
    path = tmp_path / "probe.png"
    Image.fromarray(np.full((80, 60), 128, dtype=np.uint8)).save(path)
    model, checkpoint = load_checkpoint(ORIGINAL_RUN / "selected_model.pt")
    result = predict_image(model, checkpoint, path, torch.device("cpu"), overlay_path=tmp_path / "overlay.png")
    assert result["threshold"] == .014 and result["arch"] == "simple_cnn"
    assert 0 <= result["p_defect"] <= 1
    assert (tmp_path / "overlay.png").exists()
    assert predict_image(model, checkpoint, path, torch.device("cpu"), threshold=0)["prediction"] == "defect"
    assert build_transforms(224, False)(Image.open(path)).shape == (3, 224, 224)


def test_original_outputs_are_protected():
    with pytest.raises(ValueError, match="read-only"):
        prepare_output(ORIGINAL_RUN)


def test_class_remapping_and_training_only_weights(tmp_path):
    for split, count in [("train", 20), ("test", 4)]:
        for name in ["def_front", "ok_front"]:
            target = tmp_path / split / name; target.mkdir(parents=True)
            for i in range(count): Image.fromarray(np.full((32, 32), i + 20, dtype=np.uint8)).save(target / f"{i}.png")
    cfg = load_config(); cfg["data"].update(root=str(tmp_path), image_size=32, batch_size=4, num_workers=0)
    training, validation, test, weights = build_dataloaders(cfg)
    assert [len(l.dataset) for l in [training, validation, test]] == [34, 6, 8]
    assert test.dataset[0][1] == 1  # def_front comes first alphabetically but is the positive class
    np.testing.assert_allclose(weights.numpy(), [1, 1])
