

from dataclasses import dataclass
from pathlib import Path
import pickle

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data_acc(path="data_acc"):
    """Load the trusted local pickle and return clouds and integer labels.

    Preserve group order: the 100 A clouds, then B, then C. Classes: A=0, B=1, C=2.
    """
    with Path(path).open("rb") as stream:
        data = pickle.load(stream)
    if not isinstance(data, (list, tuple)) or len(data) != 4:
        raise ValueError("data_acc doit contenir trois groupes et les labels.")
    groups = [np.asarray(group, dtype=float) for group in data[:3]]
    for group in groups:
        if group.ndim != 3 or group.shape[1:] != (200, 3):
            raise ValueError("Chaque groupe doit avoir la forme (n, 200, 3).")
        if not np.isfinite(group).all():
            raise ValueError("Les accélérations doivent être finies.")
    clouds = np.concatenate(groups, axis=0)
    labels = np.asarray(data[3])
    expected = np.concatenate([
        np.full(len(group), name) for group, name in zip(groups, "ABC")
    ])
    if labels.shape != expected.shape or not np.array_equal(labels, expected):
        raise ValueError("Les labels ne correspondent pas à l'ordre des groupes.")
    return clouds, np.searchsorted(np.array(list("ABC")), labels)


@dataclass(frozen=True)
class SplitIndices:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


def split_indices(labels, *, validation_size=0.2, test_size=0.2, random_state=42):
    """Create three stratified splits, with fractions of the full dataset."""
    if not (0 < validation_size < 1 and 0 < test_size < 1
            and validation_size + test_size < 1):
        raise ValueError("Les fractions doivent être positives et leur somme < 1.")
    labels = np.asarray(labels)
    indices = np.arange(len(labels))
    remaining, test = train_test_split(
        indices, test_size=test_size, stratify=labels, random_state=random_state
    )
    train, validation = train_test_split(
        remaining, test_size=validation_size / (1 - test_size),
        stratify=labels[remaining], random_state=random_state,
    )
    return SplitIndices(train, validation, test)


def finite_diagram(diagram):
    """Remove non-finite intervals, including essential components."""
    diagram = np.asarray(diagram, dtype=float)
    if diagram.ndim != 2 or diagram.shape[1] != 2:
        raise ValueError("Un diagramme doit avoir la forme (n, 2).")
    diagram = diagram[np.isfinite(diagram).all(axis=1)]
    if np.any(diagram[:, 1] < diagram[:, 0]):
        raise ValueError("La mort d'un intervalle ne peut précéder sa naissance.")
    return diagram


def persistence_diagram(cloud, *, max_edge_length=1.5, max_dimension=2):
    """Concatenate H0 and H1 intervals as in the notebook, without padding.

    The two coordinates do not encode the homology dimension.
    Intervals that are infinite at the max_edge_length cutoff are removed.
    """
    import gudhi

    cloud = np.asarray(cloud, dtype=float)
    if cloud.ndim != 2 or cloud.shape[1] != 3 or not np.isfinite(cloud).all():
        raise ValueError("Un nuage doit être un tableau fini de forme (n, 3).")
    if max_edge_length <= 0 or max_dimension < 2:
        raise ValueError("Il faut max_edge_length > 0 et max_dimension >= 2 pour H1.")
    tree = gudhi.RipsComplex(
        points=cloud, max_edge_length=max_edge_length
    ).create_simplex_tree(max_dimension=max_dimension)
    tree.compute_persistence(homology_coeff_field=2)
    return finite_diagram(np.concatenate([
        tree.persistence_intervals_in_dimension(dim) for dim in (0, 1)
    ]))


def compute_diagrams(clouds, **kwargs):
    return [persistence_diagram(cloud, **kwargs) for cloud in clouds]


@dataclass(frozen=True)
class DiagramBatch:
    values: np.ndarray
    mask: np.ndarray  # True = actual interval or CLS token; False = padding
    lengths: np.ndarray  # Number of intervals, excluding CLS


class DiagramPreprocessor:
    """Scale actual training points, then pad and append CLS.

    Each coordinate shares its normalization across positions to respect
    the unordered nature of diagrams. Padding and CLS are excluded from fit.
    The CLS token is appended at the last position after scaling.
    """

    def fit(self, diagrams):
        diagrams = [finite_diagram(d) for d in diagrams]
        if not diagrams or not any(len(d) for d in diagrams):
            raise ValueError("Le train doit contenir au moins un intervalle fini.")
        self.scaler_ = StandardScaler().fit(np.concatenate(diagrams))
        self.max_size_ = max(map(len, diagrams))
        return self

    def transform(self, diagrams, *, max_size=None):
        if not hasattr(self, "scaler_"):
            raise ValueError("Appeler fit sur les diagrammes du train d'abord.")
        diagrams = [finite_diagram(d) for d in diagrams]
        lengths = np.array([len(d) for d in diagrams], dtype=int)
        longest = max(lengths, default=0)
        # Expand the batch if needed: keep all validation/test intervals.
        # This size is a storage choice, not a learned statistic.
        if max_size is None:
            max_size = max(self.max_size_, longest)
        if not isinstance(max_size, (int, np.integer)) or max_size < longest:
            raise ValueError("max_size doit être un entier couvrant tous les intervalles.")
        values = np.zeros((len(diagrams), max_size + 1, 2), dtype=np.float32)
        mask = np.zeros((len(diagrams), max_size + 1), dtype=bool)
        for index, diagram in enumerate(diagrams):
            if len(diagram):
                values[index, :len(diagram)] = self.scaler_.transform(diagram)
                mask[index, :len(diagram)] = True
        values[:, -1] = 1.5
        mask[:, -1] = True
        return DiagramBatch(values, mask, lengths)


def make_atol_pipeline(*, n_clusters=40, random_state=42):
    """Apply ATOL then scaling: call fit on a list of training diagrams.

    Never pass padded diagrams to ATOL. Then call transform on validation/test;
    the centers and scaler remain those fitted on the training set.
    """
    from gudhi.representations import Atol
    from sklearn.cluster import KMeans
    from sklearn.pipeline import make_pipeline

    return make_pipeline(
        Atol(quantiser=KMeans(
            n_clusters=n_clusters, random_state=random_state, n_init=10
        )),
        StandardScaler(),
    )
