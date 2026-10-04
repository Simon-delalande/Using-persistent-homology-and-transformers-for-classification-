"""Compare logistic regression on raw accelerations and TDA features."""

import argparse
import hashlib
import json
from pathlib import Path
import warnings
from importlib.metadata import version

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler

from preprocessing import (
    compute_diagrams, load_data_acc, make_atol_pipeline, split_indices,
)


def select_logistic_regression(x_train, y_train, x_validation, y_validation):
    """Select C on validation, never on test; fit on training data only."""
    candidates = []
    best_model = None
    best_score = -np.inf
    # Break ties by keeping the smallest C (stronger regularization).
    for c in (0.001, 0.01, 0.1, 1.0, 10.0, 100.0):
        model = LogisticRegression(C=c, solver="lbfgs", max_iter=5000)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            model.fit(x_train, y_train)
        score = accuracy_score(y_validation, model.predict(x_validation))
        candidates.append({"C": c, "validation_accuracy": float(score)})
        if score > best_score:
            best_model, best_score = model, score
    return best_model, candidates


def evaluate(model, features, labels):
    predictions = model.predict(features)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "confusion_matrix": confusion_matrix(
            labels, predictions, labels=[0, 1, 2]
        ).tolist(),
        "predictions": predictions.tolist(),
    }


def load_or_compute_diagrams(clouds, cache_path):
    """Cache unpadded diagrams, validated against the data and Rips parameters."""
    parameters = {"max_edge_length": 1.5, "max_dimension": 2}
    signature = hashlib.sha256(
        np.ascontiguousarray(clouds).tobytes()
        + json.dumps(parameters, sort_keys=True).encode()
        + b"H0+H1;finite;field=2;v1"
    ).hexdigest()
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as cache:
            if str(cache["signature"]) == signature:
                return [cache[f"diagram_{i}"] for i in range(len(clouds))]
    print(f"Calcul des {len(clouds)} diagrammes de persistance...", flush=True)
    diagrams = compute_diagrams(clouds, **parameters)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path, signature=np.array(signature),
        **{f"diagram_{i}": d for i, d in enumerate(diagrams)},
    )
    return diagrams


def run_comparison(data_path="data_acc", output_dir="results/baseline", seed=42):
    """Run a paired comparison with two representations and one protocol."""
    output_dir = Path(output_dir)
    clouds, labels = load_data_acc(data_path)
    split = split_indices(labels, random_state=seed)
    indices = [split.train, split.validation, split.test]
    targets = [labels[i] for i in indices]

    raw = clouds.reshape(len(clouds), -1)
    raw_scaler = StandardScaler().fit(raw[split.train])
    raw_features = [raw_scaler.transform(raw[i]) for i in indices]

    diagrams = load_or_compute_diagrams(clouds, output_dir / "diagrams.npz")
    diagram_splits = [[diagrams[i] for i in group] for group in indices]
    atol = make_atol_pipeline(n_clusters=40, random_state=seed)
    tda_train = atol.fit_transform(diagram_splits[0])
    tda_features = [tda_train] + [atol.transform(d) for d in diagram_splits[1:]]

    report = {
        "seed": seed,
        "data_sha256": hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
        "versions": {name: version(name) for name in ("numpy", "scikit-learn", "gudhi")},
        "classes": ["A", "B", "C"],
        "split_indices": {
            name: getattr(split, name).tolist()
            for name in ("train", "validation", "test")
        },
        "test_labels": targets[2].tolist(),
        "protocol": {
            "model": "LogisticRegression(lbfgs, max_iter=5000)",
            "selection": "validation accuracy; smallest C on ties; no refit",
            "fit_scope": "train only for scaling, ATOL and classifier",
            "tda": "Rips 1.5; dimension 2; finite H0+H1; ATOL 40 centres",
        },
        "models": {},
    }
    for name, features in [("raw", raw_features), ("tda_atol", tda_features)]:
        model, candidates = select_logistic_regression(
            features[0], targets[0], features[1], targets[1]
        )
        report["models"][name] = {
            "n_features": int(features[0].shape[1]),
            "n_parameters": int(model.coef_.size + model.intercept_.size),
            "selected_C": model.C,
            "validation_search": candidates,
            **{
                group: evaluate(model, x, y)
                for group, x, y in zip(
                    ("train", "validation", "test"), features, targets
                )
            },
        }
        print(name, "C =", model.C,
              "test accuracy =", report["models"][name]["test"]["accuracy"],
              flush=True)

    # Paired counts: both models predict the same 60 samples.
    raw_correct = np.array(report["models"]["raw"]["test"]["predictions"]) == targets[2]
    tda_correct = np.array(report["models"]["tda_atol"]["test"]["predictions"]) == targets[2]
    report["paired_test_counts"] = {
        "both_correct": int(np.sum(raw_correct & tda_correct)),
        "raw_only_correct": int(np.sum(raw_correct & ~tda_correct)),
        "tda_only_correct": int(np.sum(~raw_correct & tda_correct)),
        "both_wrong": int(np.sum(~raw_correct & ~tda_correct)),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    write_summary(report, output_dir / "comparison.md")
    return report


def write_summary(report, path):
    lines = [
        "# Régression logistique : données brutes et TDA", "",
        f"Même partition stratifiée : {len(report['split_indices']['train'])} train, "
        f"{len(report['split_indices']['validation'])} validation, "
        f"{len(report['split_indices']['test'])} test (seed {report['seed']}).",
        "Normalisation, ATOL et classificateurs ajustés sur le train seulement.",
        "C choisi parmi 0.001, 0.01, 0.1, 1, 10, 100 sur la validation.", "",
        "| Représentation | Variables | Paramètres | C | Train | Validation | Test | F1 macro test |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, result in report["models"].items():
        scores = [result[group]["accuracy"] for group in ("train", "validation", "test")]
        lines.append(
            f"| {name} | {result['n_features']} | {result['n_parameters']} | "
            f"{result['selected_C']:g} | " + " | ".join(f"{s:.1%}" for s in scores)
            + f" | {result['test']['macro_f1']:.3f} |"
        )
    raw_score = report["models"]["raw"]["test"]["accuracy"]
    tda_score = report["models"]["tda_atol"]["test"]["accuracy"]
    lines += [
        "",
        f"Sur cette partition, l'écart TDA − brut est de "
        f"{100 * (tda_score - raw_score):+.1f} points de pourcentage sur le test.",
        "Le résultat concerne l'ensemble du prétraitement TDA + ATOL,",
        "pas une attribution isolée à l'homologie ou à la vectorisation.",
    ]
    for name, result in report["models"].items():
        lines += ["", f"## Matrice de confusion test : {name}", "",
                  "Lignes = vraies classes ; colonnes = prédictions.", "",
                  "| | A | B | C |", "|---|---:|---:|---:|"]
        for label, row in zip(report["classes"], result["test"]["confusion_matrix"]):
            lines.append(f"| {label} | " + " | ".join(map(str, row)) + " |")
    lines += [
        "", "## Interprétation", "",
        "Cette expérience mesure si la représentation topologique rend les classes",
        "plus faciles à séparer avec une régression logistique. Le résultat doit être",
        "lu tel quel, même si les données brutes obtiennent un score comparable ou supérieur.",
        "Les entrées ont des dimensions différentes : même famille de modèles, mais",
        "pas le même nombre de paramètres. ATOL constitue lui-même un prétraitement appris.",
        "Une partition de 60 exemples test ne démontre pas une supériorité générale.",
        "Les sessions d'acquisition ne sont pas identifiées ; la dépendance entre",
        "fenêtres ne peut donc pas être exclue par cette séparation par échantillon.",
        "Les performances de validation servent à choisir C ; elles ne sont pas",
        "une estimation indépendante des performances finales.",
    ]
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data_acc")
    parser.add_argument("--output-dir", default="results/baseline")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_comparison(args.data, args.output_dir, args.seed)
