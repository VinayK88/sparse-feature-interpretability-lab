from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TrainingResult:
    losses: list[float]
    reconstruction_losses: list[float]
    sparsity_losses: list[float]


class SparseAutoencoder:
    def __init__(self, input_dimensions: int, latent_features: int, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.encoder = rng.normal(scale=0.12, size=(input_dimensions, latent_features))
        self.decoder = rng.normal(scale=0.12, size=(latent_features, input_dimensions))
        self.bias = np.zeros(latent_features)
        self._normalize_decoder()

    def encode(self, observations: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, observations @ self.encoder + self.bias)

    def decode(self, latents: np.ndarray) -> np.ndarray:
        return latents @ self.decoder

    def reconstruct(self, observations: np.ndarray) -> np.ndarray:
        return self.decode(self.encode(observations))

    def fit(
        self,
        observations: np.ndarray,
        steps: int = 800,
        learning_rate: float = 0.020,
        l1: float = 0.040,
    ) -> TrainingResult:
        sample_count, dimensions = observations.shape
        latent_count = self.decoder.shape[0]
        losses: list[float] = []
        reconstruction_losses: list[float] = []
        sparsity_losses: list[float] = []

        # Adam keeps the tiny experiment stable across supported NumPy versions.
        parameters = [self.encoder, self.decoder, self.bias]
        first_moment = [np.zeros_like(value) for value in parameters]
        second_moment = [np.zeros_like(value) for value in parameters]
        beta1, beta2 = 0.9, 0.999

        for step in range(1, steps + 1):
            preactivation = observations @ self.encoder + self.bias
            latents = np.maximum(0.0, preactivation)
            reconstruction = latents @ self.decoder
            residual = reconstruction - observations
            reconstruction_loss = float(np.mean(residual**2))
            sparsity_loss = float(l1 * np.mean(latents))
            losses.append(reconstruction_loss + sparsity_loss)
            reconstruction_losses.append(reconstruction_loss)
            sparsity_losses.append(sparsity_loss)

            grad_reconstruction = 2.0 * residual / (sample_count * dimensions)
            grad_decoder = latents.T @ grad_reconstruction
            grad_latents = grad_reconstruction @ self.decoder.T + l1 / (sample_count * latent_count)
            grad_preactivation = grad_latents * (preactivation > 0)
            grad_encoder = observations.T @ grad_preactivation
            grad_bias = grad_preactivation.sum(axis=0)
            gradients = [grad_encoder, grad_decoder, grad_bias]

            for index, (parameter, gradient) in enumerate(zip(parameters, gradients)):
                np.clip(gradient, -2.0, 2.0, out=gradient)
                first_moment[index] = beta1 * first_moment[index] + (1 - beta1) * gradient
                second_moment[index] = beta2 * second_moment[index] + (1 - beta2) * (gradient**2)
                corrected_first = first_moment[index] / (1 - beta1**step)
                corrected_second = second_moment[index] / (1 - beta2**step)
                parameter -= learning_rate * corrected_first / (np.sqrt(corrected_second) + 1e-8)
            self._normalize_decoder()

        return TrainingResult(losses, reconstruction_losses, sparsity_losses)

    def _normalize_decoder(self) -> None:
        norms = np.linalg.norm(self.decoder, axis=1, keepdims=True)
        self.decoder /= np.maximum(norms, 1e-12)
