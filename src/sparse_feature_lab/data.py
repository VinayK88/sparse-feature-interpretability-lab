from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SyntheticFeatures:
    observations: np.ndarray
    activations: np.ndarray
    dictionary: np.ndarray
    feature_names: tuple[str, ...]


def make_superposition_dataset(
    samples: int = 1600,
    observed_dimensions: int = 12,
    feature_count: int = 18,
    sparsity: float = 0.04,
    noise: float = 0.008,
    seed: int = 17,
) -> SyntheticFeatures:
    """Mix more sparse ground-truth features than observed dimensions."""

    if feature_count <= observed_dimensions:
        raise ValueError("feature_count must exceed observed_dimensions to create superposition")
    rng = np.random.default_rng(seed)
    dictionary = rng.normal(size=(observed_dimensions, feature_count))
    dictionary /= np.linalg.norm(dictionary, axis=0, keepdims=True) + 1e-12

    active = rng.random((samples, feature_count)) < sparsity
    magnitudes = rng.uniform(0.55, 1.45, size=(samples, feature_count))
    activations = active * magnitudes

    # Add two correlated feature pairs so recovery is not artificially independent.
    for left, right in ((0, 1), (6, 7)):
        coactive = rng.random(samples) < (sparsity * 0.45)
        activations[coactive, left] = rng.uniform(0.7, 1.3, coactive.sum())
        activations[coactive, right] = rng.uniform(0.7, 1.3, coactive.sum())

    observations = activations @ dictionary.T
    observations += rng.normal(scale=noise, size=observations.shape)
    feature_names = tuple(f"feature_{index:02d}" for index in range(feature_count))
    return SyntheticFeatures(observations, activations, dictionary, feature_names)
