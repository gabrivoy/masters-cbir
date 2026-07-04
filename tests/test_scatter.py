"""Tests for the frontend scatter builder.

These guard the Plotly figure construction, in particular the 3D path: a
Scatter3d marker line width must be a scalar, not a per-point list, so
highlighting neighbours has to be done as a separate trace. A per-point list
there raises a ValueError at figure-build time (not import time), which is why
this is exercised explicitly.
"""

from __future__ import annotations

from cbir.app.streamlit_app import _scatter
from cbir.common.models import ProjectionPoint

_POINTS = [
    ProjectionPoint(item_id="a", target_class="Traineira", coords=[0.1, 0.2, 0.3]),
    ProjectionPoint(item_id="b", target_class="Rebocador", coords=[-0.1, 0.0, 0.1]),
    ProjectionPoint(item_id="c", target_class="Rebocador", coords=[0.2, -0.1, 0.05]),
]
_COLORS = {"Traineira": "#1f77b4", "Rebocador": "#ff7f0e"}


def test_scatter_3d_with_hits_and_query_builds() -> None:
    # The exact path that used to raise: 3D, with highlighted neighbours.
    fig = _scatter(
        _POINTS, query_coords=[0.15, 0.05, 0.2], hit_ids=["b", "c"],
        n_components=3, color_map=_COLORS,
    )
    # Two class traces + neighbours overlay + query.
    assert len(fig.data) == 4


def test_scatter_2d_with_hits_and_query_builds() -> None:
    fig = _scatter(
        _POINTS, query_coords=[0.15, 0.05], hit_ids=["b"],
        n_components=2, color_map=_COLORS,
    )
    assert len(fig.data) == 4


def test_scatter_without_hits_or_query_builds() -> None:
    fig = _scatter(_POINTS, query_coords=None, hit_ids=[], n_components=3, color_map=_COLORS)
    # Just the two class traces.
    assert len(fig.data) == 2
