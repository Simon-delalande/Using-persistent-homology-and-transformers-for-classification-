# Attention on persistence diagrams

Seed 42; split sizes: train=180, validation=60, test=60.
Best validation-loss epoch: 10 / 20 epochs run.
All scores below use the restored weights of that epoch.

| Split | Accuracy | Macro F1 | Loss |
|---|---:|---:|---:|
| train | 99.4% | 0.994 | 0.0360 |
| validation | 96.7% | 0.967 | 0.1588 |
| test | 96.7% | 0.967 | 0.0775 |

## Test comparison on identical partitions

| Model | Parameters | Test accuracy |
|---|---:|---:|
| Logistic regression: raw | 1803 | 91.7% |
| Logistic regression: tda_atol | 123 | 100.0% |
| Attention: TDA diagrams | 18435 | 96.7% |

## Test confusion matrix

Rows = true classes; columns = predictions.

| | A | B | C |
|---|---:|---:|---:|
| A | 19 | 1 | 0 |
| B | 0 | 19 | 1 |
| C | 0 | 0 | 20 |

The attention model receives finite H0/H1 diagram points directly, without ATOL.
Normalization is fitted on training points only. Padding is masked in every encoder.
The architecture is fixed before training; the epoch is selected on validation loss.
One sample-wise split does not establish general superiority; acquisition sessions are unknown.
