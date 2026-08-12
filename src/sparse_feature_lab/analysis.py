from __future__ import annotations

import numpy as np

from .data import SyntheticFeatures
from .model import SparseAutoencoder


def cosine_match_matrix(model: SparseAutoencoder, data: SyntheticFeatures) -> np.ndarray:
    learned = model.decoder
    truth = data.dictionary.T
    learned = learned / (np.linalg.norm(learned, axis=1, keepdims=True) + 1e-12)
    truth = truth / (np.linalg.norm(truth, axis=1, keepdims=True) + 1e-12)
    return np.abs(truth @ learned.T)


def recovery_metrics(model: SparseAutoencoder, data: SyntheticFeatures) -> dict:
    matches = cosine_match_matrix(model, data)
    best_similarity = matches.max(axis=1)
    best_latent = matches.argmax(axis=1)
    latents = model.encode(data.observations)
    active_fraction = (latents > 1e-6).mean(axis=0)
    dead_fraction = float((active_fraction < 0.002).mean())
    return {
        "mean_best_cosine": float(best_similarity.mean()),
        "median_best_cosine": float(np.median(best_similarity)),
        "recovered_at_0_80": float((best_similarity >= 0.80).mean()),
        "recovered_at_0_90": float((best_similarity >= 0.90).mean()),
        "dead_latent_fraction": dead_fraction,
        "best_similarity_by_feature": best_similarity.tolist(),
        "best_latent_by_feature": best_latent.tolist(),
    }


def causal_ablation(
    model: SparseAutoencoder,
    data: SyntheticFeatures,
    ground_truth_feature: int,
) -> dict:
    matches = cosine_match_matrix(model, data)
    latent_index = int(matches[ground_truth_feature].argmax())
    latents = model.encode(data.observations)
    reconstructed = model.decode(latents)
    ablated_latents = latents.copy()
    ablated_latents[:, latent_index] = 0.0
    ablated = model.decode(ablated_latents)

    direction = data.dictionary[:, ground_truth_feature]
    baseline_projection = reconstructed @ direction
    ablated_projection = ablated @ direction
    active = data.activations[:, ground_truth_feature] > 0
    targeted_drop = float((baseline_projection[active] - ablated_projection[active]).mean())
    inactive_drop = float(np.abs(baseline_projection[~active] - ablated_projection[~active]).mean())
    return {
        "ground_truth_feature": ground_truth_feature,
        "ablated_latent": latent_index,
        "cosine_match": float(matches[ground_truth_feature, latent_index]),
        "targeted_projection_drop": targeted_drop,
        "inactive_collateral_change": inactive_drop,
        "specificity_ratio": targeted_drop / max(inactive_drop, 1e-9),
    }
