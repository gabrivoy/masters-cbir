"""PCA projection of embeddings to 2D and 3D.

The whole point of the visualizer is to place a *new* query image in the *same*
coordinate space as the indexed gallery. PCA makes this exact and cheap: we fit
the principal components once on the gallery, then apply the identical linear
transform to any query vector. A method like t-SNE or UMAP cannot do this
transform() exactly for unseen points, which is why PCA is the right primitive
for "where does my query land relative to the clusters".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA

from cbir.config import PCA_SEED


@dataclass(frozen=True)
class ProjectionModel:
    """A fitted PCA that projects embeddings to a fixed number of components.

    Holds the fitted estimator plus the projected gallery coordinates, so the
    caller can plot the gallery and project queries without refitting.
    """

    pca: PCA
    coords: np.ndarray  # (N, n_components) gallery coordinates
    n_components: int
    explained_variance_ratio: np.ndarray

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Project new embeddings into the fitted space.

        Accepts a single (dim,) vector or an (M, dim) matrix and always returns
        a 2D (M, n_components) array.
        """
        matrix = np.atleast_2d(np.asarray(embeddings, dtype=np.float32))
        return self.pca.transform(matrix)

    @property
    def cumulative_variance(self) -> float:
        """Fraction of variance captured by the retained components."""
        return float(np.sum(self.explained_variance_ratio))


def fit_projection(embeddings: np.ndarray, n_components: int) -> ProjectionModel:
    """Fit PCA on the gallery embeddings.

    ``n_components`` is clamped to what the data can support (you cannot ask for
    more components than min(n_samples, n_features)), so a 1-item gallery or a
    request for 3D on a tiny set degrades gracefully instead of raising.
    """
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2D (N, dim) array, got shape {matrix.shape}.")
    n_samples, n_features = matrix.shape
    if n_samples == 0:
        raise ValueError("Cannot fit a projection on an empty gallery.")

    effective = min(n_components, n_samples, n_features)
    if effective < 1:
        raise ValueError("Not enough data to fit even one component.")

    pca = PCA(n_components=effective, random_state=PCA_SEED)
    coords = pca.fit_transform(matrix)

    # If the data could not support the requested dimensionality, pad the
    # coordinates with zeros so downstream 2D/3D plotting code can rely on a
    # fixed column count.
    if effective < n_components:
        pad = np.zeros((coords.shape[0], n_components - effective), dtype=coords.dtype)
        coords = np.hstack([coords, pad])
        variance = np.concatenate(
            [pca.explained_variance_ratio_, np.zeros(n_components - effective)]
        )
    else:
        variance = pca.explained_variance_ratio_

    return ProjectionModel(
        pca=pca,
        coords=coords,
        n_components=n_components,
        explained_variance_ratio=variance,
    )
