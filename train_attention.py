"""Train masked attention on TDA diagrams using the baseline data partitions."""

import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss

from attention_model import build_attention_model
from baseline import load_or_compute_diagrams
from preprocessing import DiagramPreprocessor, load_data_acc, split_indices


def make_dataset(batch, labels, *, batch_size=16, shuffle=False, seed=42):
    """Keep values, explicit masks, and integer targets aligned."""
    dataset = tf.data.Dataset.from_tensor_slices((
        {"values": batch.values, "valid_mask": batch.mask}, labels,
    ))
    if shuffle:
        dataset = dataset.shuffle(len(labels), seed=seed, reshuffle_each_iteration=True)
    options = tf.data.Options()
    options.threading.private_threadpool_size = 1
    return dataset.batch(batch_size).with_options(options).prefetch(1)


def evaluate(model, batch, labels):
    probabilities = model(
        {"values": batch.values, "valid_mask": batch.mask}, training=False,
    ).numpy()
    predictions = probabilities.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "loss": float(log_loss(labels, probabilities, labels=[0, 1, 2])),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1, 2]).tolist(),
        "predictions": predictions.tolist(),
        "probabilities": probabilities.tolist(),
    }


def run_attention(data_path="data_acc", output_dir="results/attention", seed=42,
                  epochs=50, batch_size=16, verbose=2):
    """Select the epoch by validation loss, restore it, then evaluate the test once."""
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive.")
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    clouds, labels = load_data_acc(data_path)
    split = split_indices(labels, random_state=seed)
    # Reuse only the sample-wise diagram cache, not baseline fitted transformations.
    diagrams = load_or_compute_diagrams(clouds, Path("results/baseline/diagrams.npz"))
    indices = [split.train, split.validation, split.test]
    diagram_splits = [[diagrams[i] for i in group] for group in indices]
    preprocessor = DiagramPreprocessor().fit(diagram_splits[0])
    batches = [preprocessor.transform(d) for d in diagram_splits]
    targets = [labels[i] for i in indices]
    model = build_attention_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=["accuracy"],
    )
    stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True,
    )
    history = model.fit(
        make_dataset(batches[0], targets[0], batch_size=batch_size, shuffle=True, seed=seed),
        validation_data=make_dataset(batches[1], targets[1], batch_size=batch_size),
        epochs=epochs, callbacks=[stopping, tf.keras.callbacks.TerminateOnNaN()],
        shuffle=False, verbose=verbose,
    )
    if not all(np.isfinite(values).all() for values in history.history.values()):
        raise RuntimeError("Training produced non-finite metrics; no result saved.")
    report = {
        "seed": seed,
        "data_sha256": hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
        "versions": {name: version(name) for name in ("numpy", "scikit-learn", "gudhi", "tensorflow", "keras")},
        "classes": ["A", "B", "C"],
        "split_indices": {name: getattr(split, name).tolist() for name in ("train", "validation", "test")},
        "test_labels": targets[2].tolist(),
        "n_parameters": model.count_params(),
        "protocol": {
            "architecture": "2 encoders, embedding 32, 2 heads, FFN 64, dropout 0.1, learned CLS",
            "fit_scope": "diagram scaling and model fit on train only; no ATOL",
            "mask": "explicit boolean valid-key mask in every encoder; CLS valid",
            "position_encoding": "none: diagrams are unordered sets",
            "selection": "minimum validation loss; patience 10; restore best weights",
            "learning_rate": 0.001, "max_epochs": epochs, "batch_size": batch_size,
        },
        "epochs_run": len(history.history["loss"]),
        "best_epoch": int(np.argmin(history.history["val_loss"]) + 1),
        "history": {name: [float(x) for x in values] for name, values in history.history.items()},
        "scores": {
            name: evaluate(model, batch, y)
            for name, batch, y in zip(("train", "validation", "test"), batches, targets)
        },
    }
    baseline_path = Path("results/baseline/metrics.json")
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text())
        if (baseline["data_sha256"] == report["data_sha256"]
                and baseline["split_indices"] == report["split_indices"]):
            report["baseline_scores"] = {
                name: {"test_accuracy": r["test"]["accuracy"], "n_parameters": r["n_parameters"]}
                for name, r in baseline["models"].items()
            }
    model.save(output_dir / "model.keras")
    # Save the fitted preprocessing state alongside the checkpoint for inference.
    np.savez(
        output_dir / "preprocessor.npz", mean=preprocessor.scaler_.mean_,
        scale=preprocessor.scaler_.scale_, var=preprocessor.scaler_.var_,
        n_samples_seen=preprocessor.scaler_.n_samples_seen_, max_size=preprocessor.max_size_,
    )
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    write_summary(report, output_dir / "comparison.md")
    save_learning_curves(report, output_dir / "learning_curves.png")
    print("Restored epoch:", report["best_epoch"], "Test accuracy:", report["scores"]["test"]["accuracy"], flush=True)
    return report


def write_summary(report, path):
    """Summarize the selected checkpoint and matched baseline scores."""
    lines = [
        "# Attention on persistence diagrams", "",
        f"Seed {report['seed']}; split sizes: " + ", ".join(
            f"{name}={len(indices)}" for name, indices in report["split_indices"].items()
        ) + ".",
        f"Best validation-loss epoch: {report['best_epoch']} / {report['epochs_run']} epochs run.",
        "All scores below use the restored weights of that epoch.", "",
        "| Split | Accuracy | Macro F1 | Loss |", "|---|---:|---:|---:|",
    ]
    for name, score in report["scores"].items():
        lines.append(f"| {name} | {score['accuracy']:.1%} | {score['macro_f1']:.3f} | {score['loss']:.4f} |")
    lines += ["", "## Test comparison on identical partitions", "",
              "| Model | Parameters | Test accuracy |", "|---|---:|---:|"]
    for name, score in report.get("baseline_scores", {}).items():
        lines.append(f"| Logistic regression: {name} | {score['n_parameters']} | {score['test_accuracy']:.1%} |")
    lines.append(f"| Attention: TDA diagrams | {report['n_parameters']} | {report['scores']['test']['accuracy']:.1%} |")
    lines += ["", "## Test confusion matrix", "", "Rows = true classes; columns = predictions.",
              "", "| | A | B | C |", "|---|---:|---:|---:|"]
    for name, row in zip(report["classes"], report["scores"]["test"]["confusion_matrix"]):
        lines.append(f"| {name} | " + " | ".join(map(str, row)) + " |")
    lines += ["", "The attention model receives finite H0/H1 diagram points directly, without ATOL.",
              "Normalization is fitted on training points only. Padding is masked in every encoder.",
              "The architecture is fixed before training; the epoch is selected on validation loss.",
              "One sample-wise split does not establish general superiority; acquisition sessions are unknown."]
    path.write_text("\n".join(lines) + "\n")


def save_learning_curves(report, path):
    """Plot train/validation history and mark the selected epoch."""
    from matplotlib.figure import Figure

    figure = Figure(figsize=(10, 3.5), layout="constrained")
    axes = figure.subplots(1, 2)
    epochs = range(1, report["epochs_run"] + 1)
    for axis, metric in zip(axes, ("loss", "accuracy")):
        axis.plot(epochs, report["history"][metric], label="Train")
        axis.plot(epochs, report["history"][f"val_{metric}"], label="Validation")
        axis.axvline(report["best_epoch"], color="gray", linestyle="--", label="Selected epoch")
        axis.set(xlabel="Epoch", ylabel=metric.capitalize())
        axis.legend()
    figure.savefig(path, dpi=150)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data_acc")
    parser.add_argument("--output-dir", default="results/attention")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    # Keep CPU thread use modest; tf.data also gets its own bounded pool.
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    run_attention(args.data, args.output_dir, args.seed, args.epochs, args.batch_size)
