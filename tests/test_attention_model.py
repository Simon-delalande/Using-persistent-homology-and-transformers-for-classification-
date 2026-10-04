import os
from pathlib import Path
import tempfile
import unittest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf

from attention_model import build_attention_model
from preprocessing import DiagramPreprocessor


class AttentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tf.keras.utils.set_random_seed(42)
        cls.model = build_attention_model(embedding_dim=8, feedforward_dim=16)
        cls.processor = DiagramPreprocessor().fit([
            np.array([[0., 1.], [0., 2.], [1., 3.]])
        ])

    def predict(self, batch):
        return self.model(
            {"values": batch.values, "valid_mask": batch.mask}, training=False,
        ).numpy()

    def test_padding_values_and_length_do_not_change_predictions(self):
        diagrams = [np.array([[0., 1.], [1., 3.]])]
        short = self.processor.transform(diagrams, max_size=2)
        padded = self.processor.transform(diagrams, max_size=6)
        # A nonzero padding value must also be excluded from attention.
        padded.values[~padded.mask] = 12345
        np.testing.assert_allclose(self.predict(short), self.predict(padded), atol=1e-6)

    def test_permuting_points_does_not_change_predictions(self):
        diagram = np.array([[0., 1.], [0., 2.], [1., 3.]])
        first = self.processor.transform([diagram])
        second = self.processor.transform([diagram[[2, 0, 1]]])
        np.testing.assert_allclose(self.predict(first), self.predict(second), atol=1e-6)

    def test_empty_diagram_cls_and_zero_valued_valid_points(self):
        batch = self.processor.transform([np.empty((0, 2)), np.array([[0., 1.]])])
        batch.values[1, 0] = 0
        self.assertTrue(batch.mask[1, 0])
        probabilities = self.predict(batch)
        self.assertTrue(np.isfinite(probabilities).all())
        np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-6)

    def test_serialization_preserves_predictions_and_variable_length(self):
        batch = self.processor.transform([np.array([[0., 1.]])], max_size=5)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.keras"
            self.model.save(path)
            restored = tf.keras.models.load_model(path)
            actual = restored({"values": batch.values, "valid_mask": batch.mask}).numpy()
        np.testing.assert_allclose(actual, self.predict(batch), atol=1e-6)


if __name__ == "__main__":
    unittest.main()
