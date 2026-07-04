"""Tests for the PCA projection module."""

from __future__ import annotations

import numpy as np
import pytest

from cbir.viz.projection import fit_projection


def _gallery(n: int, dim: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((n, dim)).astype(np.float32)


def test_fit_produces_requested_dimensionality() -> None:
    proj = fit_projection(_gallery(50, 16), n_components=3)
    assert proj.coords.shape == (50, 3)
    assert proj.n_components == 3


def test_transform_matches_fit_coordinates() -> None:
    # Transforming the training data must reproduce the fitted coordinates:
    # this is exactly what lets a query land in the same space as the gallery.
    gallery = _gallery(30, 12)
    proj = fit_projection(gallery, n_components=2)
    again = proj.transform(gallery)
    assert np.allclose(again, proj.coords, atol=1e-4)


def test_transform_accepts_single_vector() -> None:
    proj = fit_projection(_gallery(20, 8), n_components=3)
    out = proj.transform(_gallery(20, 8)[0])
    assert out.shape == (1, 3)


def test_degrades_when_data_cannot_support_components() -> None:
    # Only 2 samples but 3 components requested: pad with zeros, do not raise.
    proj = fit_projection(_gallery(2, 10), n_components=3)
    assert proj.coords.shape == (2, 3)
    # The padded component carries no variance.
    assert proj.explained_variance_ratio[-1] == 0.0


def test_empty_gallery_raises() -> None:
    with pytest.raises(ValueError):
        fit_projection(np.zeros((0, 8), dtype=np.float32), n_components=2)


def test_is_deterministic() -> None:
    gallery = _gallery(40, 16, seed=7)
    a = fit_projection(gallery, n_components=3)
    b = fit_projection(gallery, n_components=3)
    assert np.allclose(a.coords, b.coords)
