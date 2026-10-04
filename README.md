# Persistent homology and acceleration classification

## Step 1: inspect and prepare the data

`data_acc` is a pickle containing three groups of 100 point clouds of shape
`(200, 3)` (200 acceleration measurements with three coordinates), followed
by 300 A/B/C labels. The labels follow the group order. All acceleration values
are finite. `preprocessing.py` loads and validates this structure and encodes
A=0, B=1, C=2.

The module uses NumPy, scikit-learn, and GUDHI. Run this example from the project
root using the `.venv` environment:

```python
from preprocessing import (
    load_data_acc, split_indices, compute_diagrams, DiagramPreprocessor,
    make_atol_pipeline,
)

clouds, labels = load_data_acc()
split = split_indices(labels, random_state=42)
# 180 training, 60 validation, 60 test samples; balanced classes in each split.
diagrams = compute_diagrams(clouds)
train_diagrams = [diagrams[i] for i in split.train]
val_diagrams = [diagrams[i] for i in split.validation]
test_diagrams = [diagrams[i] for i in split.test]

preprocessor = DiagramPreprocessor().fit(train_diagrams)
train_batch = preprocessor.transform(train_diagrams)
val_batch = preprocessor.transform(val_diagrams)
test_batch = preprocessor.transform(test_diagrams)
y_train = labels[split.train]
y_val = labels[split.validation]
y_test = labels[split.test]

# Next step: vector representation followed by logistic regression.
atol = make_atol_pipeline()
features_train = atol.fit_transform(train_diagrams)
features_val = atol.transform(val_diagrams)
features_test = atol.transform(test_diagrams)
```

Diagram computation follows the notebook's Rips complex: a cutoff of 1.5,
a complex built up to dimension 2, coefficients in Z/2Z, and concatenated
finite H0 and H1 intervals. Each sample is processed independently, so the
diagrams can be cached before splitting. The H0/H1 dimension is not encoded
in the coordinates. At this cutoff, H1 intervals can also have infinite death
times: removing infinite intervals does not affect H0 alone.

Diagram standardization learns two means and two standard deviations from
actual training intervals across all positions. This respects the unordered
nature of the points. Padding and the CLS token are added afterward.
`batch.mask` is `True` for intervals and CLS, and `False` for padding.
A valid point standardized to zero remains valid: do not infer the mask from
its values. CLS is always the final token, with value `(1.5, 1.5)` after scaling.

Sequence length may differ between batches if a validation/test diagram is
longer than the training diagrams. No intervals are truncated; an explicitly
specified size that is too small raises an error. The attention model accepts
variable lengths and passes this mask to every attention layer.

ATOL receives unpadded diagrams. Its KMeans centers and feature scaling are
fitted exclusively on training data through the pipeline. For a raw acceleration
baseline, flatten the point clouds and fit a separate `StandardScaler` on
`clouds[split.train]` only. Keep the same indices for all comparisons, select
parameters on validation data, and reserve the test set for final evaluation.

Splitting is performed by sample. The file provides neither timestamps nor
session identifiers, so independence between windows from the same recording
cannot be verified here. That information would be needed for session-based
splitting. The current task identifies the three known people rather than
generalizing to new people.

## Step 2: the same simple model on raw and TDA data

Run `.venv/bin/python baseline.py` to compare two logistic regressions:

- **Raw:** the 200 × 3 acceleration values flattened into 600 features, then
  standardized.
- **TDA:** finite H0/H1 intervals vectorized with ATOL into 40 features, then
  standardized. This comparison uses neither padding nor CLS tokens.

The models use the same sample indices and solver. Each selects its
regularization parameter `C` on validation data using the same grid. Ties are
resolved by choosing the smallest `C`. The test set is not used for parameter
selection, and the models are not refitted on validation data. Transformations
and logistic regressions are fitted on the 180 training samples.

The script produces `results/baseline/comparison.md` (a table and confusion
matrices), `metrics.json` (scores, predictions, indices, versions, and the C
search), and `diagrams.npz` (an unpadded diagram cache identified by the data
and topological computation parameters). `parrot.pkl` is not used.

This comparison examines whether the TDA representation helps a linear
classifier separate the people. It does not assume that TDA will win.
The feature counts, and therefore parameter counts, differ between the models;
the result remains specific to this split and these recordings.

## Step 3: attention on persistence diagrams

Run `.venv/bin/python train_attention.py` to train the model.
`attention_training.ipynb` also provides an interactive entry point for this
step. Training uses TensorFlow/Keras and the same 180/60/60 split (seed 42) as
the logistic regressions. It reuses only the diagram cache, then fits its own
scaling on actual training points. Diagrams are passed directly to the model,
without ATOL vectorization.

Architecture in `attention_model.py`:

- Projection of the two birth/death coordinates into 32 dimensions.
- A learned CLS token replacing the final position reserved by preprocessing.
- Two attention blocks: two heads, a feed-forward network with hidden dimension
  64, residual connections, LayerNormalization, and dropout of 0.1.
- Classification from CLS through a dense layer, followed by three class
  probabilities.

No positional encoding is added. Permuting the points should leave predictions
unchanged. Each block receives the explicit valid-token mask, and padded states
are reset to zero. Sequence length is variable; CLS remains valid even for an
empty diagram. LayerNormalization is reintroduced in the projected
32-dimensional space instead of the historical notebook's two coordinates.

Adam uses a learning rate of 0.001 and batches of 16. Training is limited to
50 epochs, with early stopping after 10 epochs without improvement in validation
loss. The best epoch's weights are restored before test evaluation. Integer
labels use SparseCategoricalCrossentropy. The architecture is fixed before
training; test scores do not guide any choices.

Outputs in `results/attention/`:

- `metrics.json`: training history, selected epoch, splits, scores, and predictions.
- `comparison.md`: scores and comparisons with baselines when the data and
  indices match exactly.
- `learning_curves.png`: training/validation curves and the selected epoch.
- `model.keras` and `preprocessor.npz`: weights and scaling state for inference.

Tests check invariance to point permutations, padding values, and padding
length, as well as empty diagrams and model save/load behavior.

## Visual presentation of the results

Open `results_presentation.ipynb` with the `.venv` kernel. The notebook includes
executed outputs and presents results without retraining models:

- Protocol and training/validation/test split sizes.
- Scores for all three models and test error counts.
- Confusion matrices and sample-by-sample prediction comparisons.
- Attention learning curves with the selected epoch.
- Classifier parameter counts versus test accuracy.

It checks that both reports use the same data, indices, and labels, then exports
six PNG figures to `results/presentation/`. The size comparison excludes
preprocessing costs and ATOL centers. To refresh the presentation, run all cells
from the project root after generating the `results/*/metrics.json` reports.

## Historical notebook

`attention_TDA.ipynb` preserves the original experiments as a reference.
It standardizes data and fits ATOL before splitting, and does not mask padding.
Its old scores therefore do not evaluate the new protocol.
The current experiments use `preprocessing.py`, `baseline.py`, and
`train_attention.py`. `attention_training.ipynb` presents the corrected attention
model; the original notebook remains a historical reference.

Verification: `.venv/bin/python -m unittest discover -s tests -v`.
