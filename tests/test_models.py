"""Tests for the shared Pydantic models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from cbir.models import ClassVote, IndexResult, Prediction, SearchHit


def test_prediction_confidence_bounds_are_enforced() -> None:
    with pytest.raises(ValidationError):
        Prediction(predicted_class="x", confidence=1.5, votes=[], k=1, weighted=True)


def test_search_hit_rank_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        SearchHit(item_id="a", target_class="x", score=0.5, rank=0)


def test_index_result_allows_model_prefixed_fields() -> None:
    # Pydantic v2 protects the "model_" namespace by default; IndexResult opts
    # out so model_name / model_slug are legal field names.
    result = IndexResult(
        collection_name="c",
        model_name="openclip-vit-b-32",
        model_slug="openclip_vit_b_32_openai",
        embedding_dim=512,
        selected_records=10,
        inserted_count=10,
        collection_count=10,
    )
    assert result.model_name == "openclip-vit-b-32"
    # Round-trips through JSON.
    assert IndexResult.model_validate_json(result.model_dump_json()) == result


def test_class_vote_round_trips() -> None:
    vote = ClassVote(target_class="Traineira", count=3, weight=2.4)
    assert ClassVote.model_validate(vote.model_dump()) == vote
