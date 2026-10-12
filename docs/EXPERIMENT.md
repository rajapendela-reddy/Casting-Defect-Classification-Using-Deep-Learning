# Exact experiment specification

## Classes and preprocessing

Class 0 is acceptable, class 1 is defective. Torchvision ImageFolder assigns `def_front=0`, `ok_front=1` alphabetically, so the data wrapper flips these labels. Both training and test folders must contain those same two classes.

Training transforms, in order: grayscale conversion to three channels, resize to 224 × 224, horizontal flip (p=0.5), vertical flip (p=0.5), random affine rotation within ±15 degrees and translation fractions 0.05, tensor conversion, and ImageNet normalization.

Validation/test/prediction transforms: grayscale to three channels, resize, tensor conversion, ImageNet normalization. No augmentation is applied for these paths.

The workflow uses seed 42 for Python, NumPy, and PyTorch. cuDNN benchmarking is disabled and deterministic mode enabled; this does not guarantee identical training results across all hardware and runtime combinations.

## Split and loss

The supplied test folder stays separate. Validation indices are sampled per original folder class using `numpy.random.default_rng(42)`, `round(class_count * 0.15)`, and shuffling of the combined indices. The original manifest preserves this order.

Training class counts are 2,444 acceptable and 3,194 defective. Cross-entropy class weights use `N / (2 * class_count)`, producing approximately **1.153437** and **0.882592**. They are calculated on the optimization split only. No oversampling or synthetic image generation is used in the real experiment.

## Architectures

SimpleCNN: four 3×3 convolution blocks with channels 3→32→64→128→256, BatchNorm, ReLU, and 2×2 max pooling; adaptive average pooling; flatten; dropout 0.2; linear classifier 256→2. Parameter count: **389,890**.

ResNet18: Torchvision ImageNet `IMAGENET1K_V1` weights; dropout 0.2 plus a 512→2 linear head replaces `fc`; all parameters are fine-tuned. Parameter count: **11,177,538**.

EfficientNet-B0 exists as an optional architecture; it was not trained in the supplied run.

## Training and selection

AdamW learning rate 0.001, weight decay 0.0001. CosineAnnealingLR uses `T_max=25`. Batch size 32, two workers in the original run, maximum 25 epochs, patience 6. The loss is class-weighted cross-entropy.

A checkpoint is updated only when validation defect F1 at threshold 0.5 strictly improves. The first checkpoint achieving a tied score is retained. Architecture selection uses Python `max` over the architecture list, preserving its first entry on an exact tie.

SimpleCNN best epoch 19, stopped at epoch 25. ResNet18 best epoch 9, stopped at epoch 15. Their selected validation F1 values are equal; SimpleCNN is selected because it is listed first.

## Threshold heuristic

Only selected-model validation scores are used. The grid has 999 values from 0.001 to 0.999. Cost is 10×FN + 1×FP. A seeded bootstrap with 200 samples estimates the standard deviation of resampled minimum costs. Accept grid positions with original cost ≤ minimum cost + that standard deviation; find the widest contiguous accepted interval and select its midpoint.

This is a one-standard-error-style heuristic. It does not implement a formal hypothesis test or prove that the accepted thresholds are statistically equivalent.

Original results: argmin 0.015, minimum cost 8, bootstrap standard error 3.5978636877, accepted widest interval [0.012, 0.016], midpoint 0.014. The selected midpoint's validation cost is 11; the tolerance is 11.5978636877.

## Inference and explainability

Softmax output index 1 gives the raw defect score. Predicted defect means score ≥ chosen threshold. `selected_model.pt` stores 0.014; raw architecture checkpoints do not store this operating point and default to 0.5 in the prediction CLI unless overridden.

Grad-CAM explains class index 1 explicitly, using the last convolutional block and gradients of the defect logit. A forward hook captures activations; a tensor hook captures their gradients. Gradient weights are spatial means; the weighted activation sum passes through ReLU and bilinear resizing. The heatmap is normalized for display, then blended with the image.

## Repository adaptation

The original completed notebook is retained. The six core source modules originate from its exported source snapshot. Working-module documentation is edited to describe this project without inheriting unrelated thesis statements; executable model/transform/metric/threshold behavior is preserved. Added CLI modules reorganize the notebook's training, evaluation, and single-image prediction functions and protect original outputs.

The included original artifacts have no substituted scores. New training runs save their own metadata and reports. Single-checkpoint evaluation is recorded as `provided_checkpoint`, rather than claiming new architecture selection.
