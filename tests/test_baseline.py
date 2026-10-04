import unittest

import numpy as np

from baseline import evaluate, select_logistic_regression


class BaselineTests(unittest.TestCase):
    def test_validation_selects_stronger_regularization_on_ties(self):
        train = np.array([[-3., 0.], [-2., 0.], [0., 3.],
                          [0., 2.], [3., 0.], [2., 0.]])
        labels = np.array([0, 0, 1, 1, 2, 2])
        model, candidates = select_logistic_regression(
            train, labels, train / 2, labels
        )
        self.assertEqual(model.C, 0.001)
        self.assertEqual(len(candidates), 6)
        self.assertTrue(all(c['validation_accuracy'] == 1 for c in candidates))
        result = evaluate(model, train, labels)
        self.assertEqual(result['accuracy'], 1)
        self.assertEqual(result['macro_f1'], 1)
        self.assertEqual(result['confusion_matrix'], [[2, 0, 0], [0, 2, 0], [0, 0, 2]])


if __name__ == "__main__":
    unittest.main()
