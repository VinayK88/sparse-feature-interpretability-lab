from __future__ import annotations

import unittest

import numpy as np

from sparse_feature_lab.analysis import causal_ablation, recovery_metrics
from sparse_feature_lab.data import make_superposition_dataset
from sparse_feature_lab.model import SparseAutoencoder


class InterpretabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = make_superposition_dataset(samples=650, seed=17)
        cls.model = SparseAutoencoder(12, 36, seed=3)
        cls.training = cls.model.fit(cls.data.observations, steps=350)

    def test_dataset_has_superposition(self) -> None:
        self.assertGreater(self.data.activations.shape[1], self.data.observations.shape[1])

    def test_training_reduces_objective(self) -> None:
        self.assertLess(self.training.losses[-1], self.training.losses[0] * 0.55)

    def test_decoder_directions_are_normalized(self) -> None:
        norms = np.linalg.norm(self.model.decoder, axis=1)
        self.assertTrue(np.allclose(norms, 1.0, atol=1e-6))

    def test_recovery_metrics_are_bounded(self) -> None:
        metrics = recovery_metrics(self.model, self.data)
        self.assertGreater(metrics["mean_best_cosine"], 0.70)
        self.assertLessEqual(metrics["mean_best_cosine"], 1.0)
        self.assertGreater(metrics["recovered_at_0_80"], 0.30)

    def test_causal_ablation_changes_target_projection(self) -> None:
        result = causal_ablation(self.model, self.data, 0)
        self.assertGreater(result["targeted_projection_drop"], 0.0)
        self.assertGreater(result["specificity_ratio"], 1.0)


if __name__ == "__main__":
    unittest.main()
