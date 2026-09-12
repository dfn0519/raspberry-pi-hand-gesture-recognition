# Model analysis

## Why hand landmarks

The initial baseline resized each image to 64 × 64 grayscale pixels and flattened it into a 4,096-dimensional vector. This approach retained background and lighting information that was unrelated to the gesture.

The improved pipeline uses MediaPipe to extract 21 hand landmarks. Each landmark contributes x, y, and z coordinates, producing a 63-dimensional feature vector. Two normalization steps are then applied:

1. Translation normalization subtracts the wrist coordinate from every landmark.
2. Scale normalization divides all coordinates by the largest landmark distance from the wrist.

This makes the representation less sensitive to where the hand appears in the frame and how far it is from the camera. In the original experiment, scale normalization improved the SVM accuracy from 88.1% to 93.8%.

## Evaluation results

All models were evaluated on the same held-out set of 369 samples.

| Model | Accuracy | Precision | Recall | F1-score |
| --- | ---: | ---: | ---: | ---: |
| SVM (RBF) | 93.77% | 94.50% | 93.77% | 93.64% |
| Random Forest | 77.24% | 86.56% | 77.24% | 74.68% |
| MLP Neural Network | 74.80% | 85.75% | 74.80% | 70.73% |

The SVM produced the most balanced result. Rock achieved perfect recall but a lower precision of 0.86, while Paper achieved perfect precision but a recall of 0.81. Scissors was the most stable class, with both precision and recall above 0.97.

Random Forest and MLP performed well on Scissors but frequently confused Paper with another class. With this dataset size and compact geometric representation, the RBF SVM learned a better decision boundary than the two more complex alternatives.

## Deployment decision

SVM was selected because it combined the best test metrics with a small serialized model and fast inference. These properties fit the limited compute resources of a Raspberry Pi.

During live inference, a prediction is accepted only when its highest class probability exceeds 0.70. Lower-confidence inputs are displayed as `Unknown`. This reduces forced classifications for unsupported gestures, although the threshold was chosen as an engineering setting rather than through a separate calibration experiment.

## Limitations and next steps

- The reported results come from one held-out split; cross-validation would provide a stronger estimate of generalization.
- The dataset is not included, so the historical metrics cannot be independently reproduced from this repository alone.
- Accuracy and model size were considered, but inference latency and frames per second were not recorded systematically.
- Future testing should include multiple users, lighting conditions, backgrounds, hand orientations, and camera distances.
- Confusion matrices and per-device latency would make the deployment comparison more complete.
