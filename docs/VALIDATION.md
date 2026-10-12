# Repository validation

Validated on **2026-10-12** in a local CPU environment with Python 3.12.14.

## Completed checks

- **11 regression tests passed**, covering stored evidence, checkpoint compatibility, AUC tie handling, class convention, threshold selection, all three supported architecture/Grad-CAM combinations, prediction preprocessing, class remapping, and output protection.
- Recalculated both original test reports from `test_predictions.npz` and confirmed the CSV alignment: threshold 0.5 gives TN=262, FP=0, FN=3, TP=450; threshold 0.014 gives TN=252, FP=10, FN=0, TP=453.
- Recreated the original class-wise split and its ordering from the supplied manifest and seed.
- Reproduced the validation bootstrap threshold of **0.014**, the accepted interval **[0.012, 0.016]**, and its reported standard error from the saved validation predictions.
- Loaded original SimpleCNN and ResNet18 checkpoints with `weights_only=True`, rebuilt their architectures without pretrained downloads, and checked finite two-class forward outputs.
- Confirmed `selected_model.pt` has exactly the same state_dict tensor values as the best SimpleCNN checkpoint, with epoch 19 and operating threshold 0.014.
- Verified parameter counts: SimpleCNN **389,890**, ResNet18 **11,177,538**.
- Executed a complete small synthetic-data run: trained SimpleCNN and non-pretrained ResNet18 for one epoch, selected by validation F1, chose a threshold, produced evaluation plots/Grad-CAM, and saved/verified the new checkpoints and reports.
- Ran the standalone evaluation CLI on that synthetic checkpoint and confirmed it reports `provided_checkpoint` selection.
- Ran the prediction CLI and saved a Grad-CAM overlay using the original selected checkpoint with a synthetic probe image. This checks inference plumbing, not real-dataset accuracy.
- Validated notebook format and code syntax, working-source syntax, workflow YAML, local documentation links, original artifact hashes, and ZIP integrity.

The synthetic data and their metrics are validation intermediates only and are not included as project benchmark results.

## Tested package versions

| Package | Local version |
|---|---|
| PyTorch | 2.11.0+cpu |
| Torchvision | 0.26.0+cpu |
| NumPy | 2.3.5 |
| Matplotlib | 3.10.8 |
| Pillow | 12.3.0 |
| PyYAML | 6.0.3 |
| pytest | 9.1.1 |
| nbformat | 5.11.1 |

## Limits of verification

The original source image folders were not supplied. Full real-image inference and retraining on the original casting dataset were therefore not repeated during repository preparation. Verification reproduces numerical reports from saved probabilities and checks checkpoint loading/structure; it does not independently recompute every original image probability.

The original notebook is included and validated structurally; it was not re-executed in Kaggle during this task. GitHub Actions is configured but has not been run in a hosted repository. GPU results and retraining can differ across runtime/hardware versions.
