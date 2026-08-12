from __future__ import annotations

from statistics import mean, pstdev

import numpy as np

from .analysis import causal_ablation, recovery_metrics
from .data import make_superposition_dataset
from .model import SparseAutoencoder


def run_experiment(samples: int = 2400, steps: int = 1200, seeds: tuple[int, ...] = (3, 11, 29)) -> dict:
    data = make_superposition_dataset(samples=samples)
    runs = []
    for seed in seeds:
        model = SparseAutoencoder(data.observations.shape[1], latent_features=36, seed=seed)
        training = model.fit(data.observations, steps=steps)
        recovery = recovery_metrics(model, data)
        interventions = [causal_ablation(model, data, feature) for feature in (0, 5, 11, 17)]
        runs.append(
            {
                "seed": seed,
                "initial_loss": training.losses[0],
                "final_loss": training.losses[-1],
                "loss_reduction": 1.0 - training.losses[-1] / training.losses[0],
                "recovery": recovery,
                "interventions": interventions,
                "loss_curve": training.losses[:: max(1, steps // 40)],
            }
        )

    mean_cosines = [item["recovery"]["mean_best_cosine"] for item in runs]
    recovered = [item["recovery"]["recovered_at_0_80"] for item in runs]
    specificity = [
        intervention["specificity_ratio"]
        for item in runs
        for intervention in item["interventions"]
    ]
    return {
        "experiment": "sparse-feature-interpretability-v0.1",
        "dataset": {
            "samples": samples,
            "observed_dimensions": data.observations.shape[1],
            "ground_truth_features": data.activations.shape[1],
            "latent_features": 36,
            "training_steps": steps,
        },
        "summary": {
            "mean_best_cosine": mean(mean_cosines),
            "seed_std_best_cosine": pstdev(mean_cosines),
            "mean_recovered_at_0_80": mean(recovered),
            "median_intervention_specificity": float(np.median(specificity)),
        },
        "runs": runs,
    }
