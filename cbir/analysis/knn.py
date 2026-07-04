"""KNN class prediction from retrieval neighbours.

Given the top-k Milvus neighbours of a query embedding, decide which class the
query would be assigned to. This is the retrieval-based auto-labeling heuristic
the thesis is ultimately about, in its simplest form: a vote over the nearest
neighbours, either by raw count or weighted by similarity.

Kept pure and dependency-free (no Milvus, no torch) so it is trivial to unit
test with synthetic neighbour lists.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from cbir.common.models import ClassVote, Prediction, SearchHit


def predict_class(
    hits: Sequence[SearchHit],
    *,
    k: int | None = None,
    weighted: bool = True,
) -> Prediction:
    """Predict the query's class by voting over its nearest neighbours.

    Args:
        hits: retrieval neighbours, already ranked best-first.
        k: how many neighbours to vote with; ``None`` uses all provided hits.
        weighted: when True, each neighbour contributes its cosine similarity;
            when False, each contributes 1 (plain majority vote).

    Returns:
        A :class:`Prediction`. ``predicted_class`` is ``None`` for an empty
        neighbour set. Confidence is the winning class's share of the total
        vote mass (weighted or count), so it lies in ``[0, 1]``.

    Ties are broken deterministically: higher vote mass, then higher count,
    then class name, so the same neighbours always yield the same answer.
    """
    considered = list(hits) if k is None else list(hits)[:k]
    effective_k = len(considered)

    if not considered:
        return Prediction(predicted_class=None, confidence=0.0, votes=[], k=0, weighted=weighted)

    counts: dict[str, int] = defaultdict(int)
    weights: dict[str, float] = defaultdict(float)
    for hit in considered:
        counts[hit.target_class] += 1
        # Clamp similarity to a non-negative weight; a negative cosine should
        # not subtract evidence, it should simply not add any.
        weights[hit.target_class] += max(0.0, hit.score)

    votes = [
        ClassVote(target_class=cls, count=counts[cls], weight=weights[cls])
        for cls in counts
    ]
    mass_of = (lambda v: v.weight) if weighted else (lambda v: float(v.count))
    # Sort strongest first: higher mass, then higher count, then class name
    # ascending. Numeric keys are negated so the name tiebreak stays A->Z
    # instead of being reversed along with them.
    votes.sort(key=lambda v: (-mass_of(v), -v.count, v.target_class))

    total_mass = sum(mass_of(v) for v in votes)
    winner = votes[0]
    confidence = (mass_of(winner) / total_mass) if total_mass > 0 else 0.0

    return Prediction(
        predicted_class=winner.target_class,
        confidence=confidence,
        votes=votes,
        k=effective_k,
        weighted=weighted,
    )
