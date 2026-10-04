import unittest

import numpy as np

from preprocessing import (
    DiagramPreprocessor, finite_diagram, make_atol_pipeline, split_indices,
)


class PreprocessingTests(unittest.TestCase):
    def test_split_is_disjoint_stratified_and_reproducible(self):
        labels = np.repeat(np.arange(3), 100)
        split = split_indices(labels)
        repeated = split_indices(labels)
        np.testing.assert_array_equal(
            np.sort(np.concatenate([split.train, split.validation, split.test])),
            np.arange(300),
        )
        for name, count in [("train", 60), ("validation", 20), ("test", 20)]:
            indices = getattr(split, name)
            np.testing.assert_array_equal(indices, getattr(repeated, name))
            np.testing.assert_array_equal(
                np.bincount(labels[indices]), [count] * 3
            )

    def test_scaling_excludes_padding_cls_and_held_out_points(self):
        train = [np.array([[0., 2.], [2., 4.]]), np.empty((0, 2))]
        processor = DiagramPreprocessor().fit(train)
        np.testing.assert_allclose(processor.scaler_.mean_, [1., 3.])
        batch = processor.transform(train)
        np.testing.assert_allclose(batch.values[0, :2], [[-1, -1], [1, 1]])
        np.testing.assert_array_equal(batch.values[1, :-1], 0)
        np.testing.assert_array_equal(batch.mask, [[True, True, True],
                                                   [False, False, True]])
        np.testing.assert_array_equal(batch.values[:, -1], 1.5)
        processor.transform([np.array([[100., 200.]])])
        np.testing.assert_allclose(processor.scaler_.mean_, [1., 3.])

    def test_longer_held_out_diagrams_are_not_truncated(self):
        processor = DiagramPreprocessor().fit([np.array([[0., 1.]])])
        held_out = [np.array([[0., 1.], [1., 2.], [2., 3.]])]
        batch = processor.transform(held_out)
        self.assertEqual(batch.values.shape, (1, 4, 2))
        self.assertTrue(batch.mask.all())
        with self.assertRaises(ValueError):
            processor.transform(held_out, max_size=1)

    def test_filter_non_finite_intervals(self):
        diagram = np.array([[0, 1], [0, np.inf], [np.nan, 2]])
        np.testing.assert_array_equal(finite_diagram(diagram), [[0, 1]])
        self.assertEqual(finite_diagram(np.empty((0, 2))).shape, (0, 2))
        with self.assertRaises(ValueError):
            finite_diagram([[2, 1]])

    def test_atol_transform_does_not_refit(self):
        train = [np.array([[0., 1.], [0., 2.]]),
                 np.array([[1., 3.], [2., 4.]])]
        pipeline = make_atol_pipeline(n_clusters=2)
        features = pipeline.fit_transform(train)
        atol = pipeline.steps[0][1]
        centers = atol.quantiser.cluster_centers_.copy()
        mean = pipeline.steps[1][1].mean_.copy()
        result = pipeline.transform([np.array([[100., 200.]])])
        self.assertEqual(features.shape, (2, 2))
        self.assertEqual(result.shape, (1, 2))
        np.testing.assert_array_equal(atol.quantiser.cluster_centers_, centers)
        np.testing.assert_array_equal(pipeline.steps[1][1].mean_, mean)


if __name__ == "__main__":
    unittest.main()
