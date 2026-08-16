# Evaluation Results

**These are real numbers from an actual run on this machine (CPU only, no
GPU available) — not estimates or numbers copied from the original
Kaggle/GPU coursework project.**

## Run configuration

- Device: CPU (no GPU on this machine)
- Image resolution: 128x128
- Checkpoint: `checkpoints/best_model.pt` (epoch 8, selected by best `val_f1`: val_loss=0.7111, val_acc=0.6512, val_f1=0.6809)
- Model: EmergencyVehicleCNN, base_channels=16, dropout=0.5
- Train / val / test sizes: 205 / 43 / 45
- Epochs trained: 40
- Training wall-clock time: 259.9s
- PyTorch: 2.2.2+cpu | Python: 3.9.13

## Test set metrics

| Metric | Value |
|---|---|
| Accuracy | 0.5778 |
| Precision | 0.5652 |
| Recall | 0.5909 |
| F1-score | 0.5778 |
| ROC-AUC | 0.6482 |

## Confusion matrix

Rows = true label, columns = predicted label. Class order: `['non_emergency', 'emergency']`

|  | Pred: non_emergency | Pred: emergency |
|---|---|---|
| True: non_emergency | 13 | 10 |
| True: emergency | 9 | 13 |

## Training behavior (honesty note)

Full per-epoch history is in `results/training_log.json`. With only ~205
training images and a ~1.1M-parameter CNN, training accuracy climbs past
90% within ~20-30 epochs while validation accuracy plateaus/oscillates in
the 0.44-0.68 range — classic overfitting on a small dataset. The checkpoint
above was selected as the epoch with the best validation F1 (not simply the
final epoch), which is why it comes from partway through training rather
than the end.

## Comparison to the original coursework project

The original LinkedIn-described project reported **81.85% test accuracy**,
trained on a Kaggle dataset using a T4/P100 GPU. This rebuild measured
**57.78% accuracy** on a much smaller, CPU-feasible
Wikimedia-sourced dataset (45 test images vs. a presumably much
larger Kaggle set) at 128x128 resolution. The gap is
attributable to: (1) a ~150x smaller dataset (293 total images here vs. a
typical Kaggle emergency-vehicle dataset in the thousands), (2) images
pulled from heterogeneous real-world Commons photos rather than a
purpose-curated Kaggle set, (3) lower resolution and a narrower network
sized to avoid immediately memorizing ~205 training images, and (4) no GPU,
which bounded how much architecture/hyperparameter search was practical.
See the README Limitations section for further discussion — this delta is
expected, not a discrepancy to be hidden.
