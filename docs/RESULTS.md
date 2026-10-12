# Interpreting the supplied results

## Model comparison

The two trained architectures reached the same best validation defect F1: 0.9982238011. SimpleCNN's best checkpoint is epoch 19; ResNet18's is epoch 9. Their validation accuracy, precision, and recall also match at those checkpoints. SimpleCNN's validation ROC-AUC is 0.9999465205; ResNet18's is 0.9998354478.

Selection uses validation F1, so SimpleCNN wins by list order on the tie. This is not evidence that it is statistically better. ResNet18 reached the tied score earlier. Test results were exported only for SimpleCNN.

## SimpleCNN test results

At threshold 0.5:

| Actual / predicted | Acceptable | Defective |
|---|---:|---:|
| Acceptable | 262 | 0 |
| Defective | 3 | 450 |

712/715 correct = **99.5804% accuracy**; **0.9966777 F1**; **99.3377% defect recall**.

At the validation-selected threshold 0.014:

| Actual / predicted | Acceptable | Defective |
|---|---:|---:|
| Acceptable | 252 | 10 |
| Defective | 0 | 453 |

705/715 correct = **98.6014% accuracy**; **0.9890830 F1**; **100% defect recall**, with **10 false alarms**.

Both have ROC-AUC 1.0 because AUC evaluates score ranking rather than a chosen threshold. Perfect ranking does not make either operating point error-free.

## Assumed cost

With FN cost 10 and FP cost 1, the default cost is 30 and the tuned cost is 10: a 66.7% reduction in this assumed objective. No actual factory costs, saved revenue, or production outcomes were measured.

The tuned interval is narrow, [0.012, 0.016], so the result should not be described as proof of threshold robustness across cameras, lighting conditions, or production lines.

## Grad-CAM

The supplied visualizations explain the defect logit. Correctly classified samples highlight ring/surface regions. False alarms also activate casting regions and sometimes background regions. This suggests areas for inspection, but does not establish a causal reason for an error or verified defect localization. Six of ten false alarms appear in the grid.

## Example prediction

The stored demonstration image is `test/def_front/cast_def_0_1059.jpeg`, with raw defect score **0.9999817609786987**, classified defective at threshold 0.014. It is a test example, not additional independent validation. Its original full-resolution pixels were not included in the output archive.

## Supported resume statement

Trained a custom CNN and ImageNet-pretrained ResNet18 on 5,638 images for binary casting defect classification. The custom CNN achieved 99.58% accuracy and 0.9967 F1 on 715 test images at threshold 0.5. Added Grad-CAM and validation-based threshold selection, reaching 100% defect recall with 10 false alarms at threshold 0.014.

Do not attribute the SimpleCNN test score to ResNet18, combine default accuracy with tuned recall as one result, claim EfficientNet training results, or describe the classification heatmaps as verified segmentation.
