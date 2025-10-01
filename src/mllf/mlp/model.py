"""Simple fully-connected MLP implemented with NumPy.

This is a lightweight, dependency-free implementation intended for small
experiments and unit tests. It supports variable-sized input vectors that can
concatenate atomic features and a solvent-status vector.
"""

from __future__ import annotations

import numpy as np
from typing import List, Callable, Optional


def _relu(x: np.ndarray) -> np.ndarray:
	return np.maximum(0, x)


def _identity(x: np.ndarray) -> np.ndarray:
	return x


class SimpleMLP:
	"""A tiny MLP with one or more hidden layers.

	This class is NOT optimized for performance. It's a small, test-friendly
	implementation that supports fit via simple gradient descent.

	Parameters
	- layers: list of ints including input dim and output dim, e.g. [in_dim, 64, 32, out_dim]
	- activation: callable for hidden layers (default ReLU)
	- lr: learning rate for gradient descent
	- seed: optional random seed
	"""

	def __init__(self, layers: List[int], activation: Callable = _relu, lr: float = 1e-3, seed: Optional[int] = None):
		assert len(layers) >= 2, "layers must include input and output sizes"
		self.layers = layers
		self.activation = activation
		self.lr = lr
		self.rng = np.random.default_rng(seed)

		# initialize weights and biases
		self.weights = [self.rng.normal(scale=0.1, size=(layers[i], layers[i + 1])) for i in range(len(layers) - 1)]
		self.biases = [np.zeros((layers[i + 1],), dtype=float) for i in range(len(layers) - 1)]

	def forward(self, X: np.ndarray) -> np.ndarray:
		h = X
		for i, (W, b) in enumerate(zip(self.weights, self.biases)):
			h = h @ W + b
			if i < len(self.weights) - 1:
				h = self.activation(h)
		return h

	def predict(self, X: np.ndarray) -> np.ndarray:
		return self.forward(X)

	def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 10, batch_size: int = 32):
		# Very small and naive gradient descent for demonstration only.
		n = X.shape[0]
		for epoch in range(epochs):
			# simple SGD
			indices = self.rng.permutation(n)
			for start in range(0, n, batch_size):
				batch_idx = indices[start:start + batch_size]
				xb = X[batch_idx]
				yb = y[batch_idx]

				# forward pass
				activations = [xb]
				pre_acts = []
				h = xb
				for i, (W, b) in enumerate(zip(self.weights, self.biases)):
					z = h @ W + b
					pre_acts.append(z)
					if i < len(self.weights) - 1:
						h = self.activation(z)
					else:
						h = z
					activations.append(h)

				# mean squared error gradient at output
				preds = activations[-1]
				grad = (2.0 / preds.shape[0]) * (preds - yb)

				# backward pass (very naive)
				for i in reversed(range(len(self.weights))):
					a_prev = activations[i]
					W = self.weights[i]

					# gradient wrt weights/biases
					dW = a_prev.T @ grad
					db = np.sum(grad, axis=0)

					# update
					self.weights[i] -= self.lr * dW
					self.biases[i] -= self.lr * db

					# propagate gradient to previous layer if not input
					if i > 0:
						if self.activation is _relu:
							da = grad @ W.T
							dz = da * (pre_acts[i - 1] > 0)
						else:
							dz = grad @ W.T
						grad = dz

	def score_mse(self, X: np.ndarray, y: np.ndarray) -> float:
		preds = self.predict(X)
		return float(np.mean((preds - y) ** 2))

