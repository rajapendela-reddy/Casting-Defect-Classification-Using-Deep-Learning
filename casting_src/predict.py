"""Classify one image and optionally save a Grad-CAM overlay; no service required."""
import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from .checkpoints import load_checkpoint
from .common import ORIGINAL_RUN, ROOT, resolve_device
from .data import CLASS_NAMES, build_transforms
from .gradcam import GradCAM, denormalize, overlay_heatmap
from .model import get_target_layer


def predict_image(model, checkpoint, image_path, device, threshold=None, overlay_path=None):
    if threshold is None:
        threshold = float(checkpoint.get("operating_threshold", 0.5))
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must lie between 0 and 1.")
    size = checkpoint["config"]["data"]["image_size"]
    with Image.open(image_path) as image:
        x = build_transforms(size, train=False)(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        p = float(torch.softmax(model(x), dim=1)[0, 1].item())
    result = {"image": str(image_path), "prediction": CLASS_NAMES[int(p >= threshold)],
              "p_defect": p, "threshold": threshold, "arch": checkpoint["arch"]}
    if overlay_path is not None:
        cam_engine = GradCAM(model, get_target_layer(model, checkpoint["arch"]))
        try:
            with torch.enable_grad():
                cam, _, _ = cam_engine(x, class_idx=1)
        finally:
            cam_engine.remove()
        overlay = overlay_heatmap(denormalize(x[0]), cam)
        overlay_path = Path(overlay_path)
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray((overlay * 255).astype("uint8")).save(overlay_path)
        result["gradcam_overlay"] = str(overlay_path)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=ORIGINAL_RUN / "selected_model.pt")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--overlay", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=ROOT / "predictions/prediction.json")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(ORIGINAL_RUN.resolve()) or (
        args.overlay and args.overlay.resolve().is_relative_to(ORIGINAL_RUN.resolve())):
        raise ValueError("Choose prediction outputs outside the original artifact folder.")
    device = resolve_device(args.device)
    model, ckpt = load_checkpoint(args.checkpoint, device)
    result = predict_image(model, ckpt, args.image, device, args.threshold, args.overlay)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
