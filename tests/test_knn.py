"""Tests for the KNN class-prediction heuristic."""

from __future__ import annotations

from cbir.knn import predict_class
from cbir.models import SearchHit


def _hit(item_id: str, cls: str, score: float, rank: int) -> SearchHit:
    return SearchHit(item_id=item_id, target_class=cls, score=score, rank=rank)


def test_empty_neighbours_yields_no_prediction() -> None:
    pred = predict_class([], weighted=True)
    assert pred.predicted_class is None
    assert pred.confidence == 0.0
    assert pred.votes == []


def test_unanimous_neighbours_full_confidence() -> None:
    hits = [_hit("a", "Traineira", 0.9, 1), _hit("b", "Traineira", 0.8, 2)]
    pred = predict_class(hits, weighted=False)
    assert pred.predicted_class == "Traineira"
    assert pred.confidence == 1.0


def test_majority_vote_ignores_similarity_when_unweighted() -> None:
    # Two weak Traineira neighbours beat one strong Lancha by count.
    hits = [
        _hit("a", "Traineira", 0.30, 1),
        _hit("b", "Traineira", 0.31, 2),
        _hit("c", "Lancha", 0.99, 3),
    ]
    pred = predict_class(hits, weighted=False)
    assert pred.predicted_class == "Traineira"
    assert pred.confidence == 2 / 3


def test_weighted_vote_can_flip_the_winner() -> None:
    # Same neighbours, but weighting by similarity lets the strong Lancha win.
    hits = [
        _hit("a", "Traineira", 0.30, 1),
        _hit("b", "Traineira", 0.31, 2),
        _hit("c", "Lancha", 0.99, 3),
    ]
    pred = predict_class(hits, weighted=True)
    assert pred.predicted_class == "Lancha"
    # confidence = 0.99 / (0.30 + 0.31 + 0.99)
    assert abs(pred.confidence - 0.99 / 1.60) < 1e-6


def test_k_limits_the_neighbours_considered() -> None:
    hits = [
        _hit("a", "Traineira", 0.9, 1),
        _hit("b", "Lancha", 0.8, 2),
        _hit("c", "Lancha", 0.7, 3),
    ]
    pred = predict_class(hits, k=1, weighted=False)
    assert pred.k == 1
    assert pred.predicted_class == "Traineira"
    assert pred.confidence == 1.0


def test_negative_similarity_does_not_subtract_evidence() -> None:
    hits = [_hit("a", "Traineira", -0.5, 1), _hit("b", "Traineira", 0.5, 2)]
    pred = predict_class(hits, weighted=True)
    # The negative-similarity neighbour contributes zero weight, not -0.5.
    winner = next(v for v in pred.votes if v.target_class == "Traineira")
    assert winner.weight == 0.5


def test_ties_are_broken_deterministically() -> None:
    hits = [_hit("a", "B", 0.5, 1), _hit("b", "A", 0.5, 2)]
    pred = predict_class(hits, weighted=True)
    # Equal count and weight -> class name breaks the tie ("A" before "B").
    assert pred.predicted_class == "A"
