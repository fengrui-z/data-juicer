import json

import pytest

from data_juicer.core.elasticjuicer.mode import ElasticJuicerMode
from data_juicer.core.elasticjuicer.scheduler.static_recommender import (
    StaticBatchRecommender,
)
from data_juicer.core.executor.default_executor import (
    _run_static_batch_recommendation,
)


class FakeOp:
    def __init__(self, name, batch_size, batched=True):
        self._name = name
        self.batch_size = batch_size
        self._batched = batched

    def is_batched_op(self):
        return self._batched


class FakeAdapter:
    def __init__(self, candidates):
        self.candidates = candidates
        self.calls = 0

    def adapt_workloads(self, dataset, operators):
        self.calls += 1
        return self.candidates


def test_recommender_marks_non_batched_operators_ineligible(tmp_path):
    operators = [FakeOp("mapper", 4), FakeOp("selector", 1, batched=False)]
    recommender = StaticBatchRecommender(str(tmp_path))

    recommendations = recommender.recommend(operators, [16, 99])

    assert recommendations[0].recommended_batch_size == 16
    assert recommendations[0].eligible
    assert recommendations[1].recommended_batch_size == 1
    assert not recommendations[1].eligible


def test_recommend_mode_does_not_mutate_operators(tmp_path):
    operators = [FakeOp("mapper", 4)]
    adapter = FakeAdapter([16])
    recommender = StaticBatchRecommender(str(tmp_path))

    _run_static_batch_recommendation(
        object(),
        operators,
        adapter,
        recommender,
        ElasticJuicerMode.RECOMMEND,
    )

    assert operators[0].batch_size == 4
    assert adapter.calls == 1


def test_apply_mode_updates_only_eligible_operators(tmp_path):
    operators = [FakeOp("mapper", 4), FakeOp("selector", 1, batched=False)]
    adapter = FakeAdapter([16, 99])
    recommender = StaticBatchRecommender(str(tmp_path))

    _run_static_batch_recommendation(
        object(),
        operators,
        adapter,
        recommender,
        ElasticJuicerMode.APPLY,
    )

    assert [operator.batch_size for operator in operators] == [16, 1]


def test_recommendations_are_persisted_atomically(tmp_path):
    recommender = StaticBatchRecommender(str(tmp_path))
    recommender.recommend([FakeOp("mapper", 4)], [8])

    data = json.loads((tmp_path / "batch_recommendations.json").read_text())
    assert data[0]["stage_name"] == "mapper"
    assert data[0]["recommended_batch_size"] == 8
    assert not (tmp_path / "batch_recommendations.tmp").exists()


def test_candidate_count_must_match_operators():
    recommender = StaticBatchRecommender()
    with pytest.raises(ValueError, match="must match"):
        recommender.recommend([FakeOp("mapper", 4)], [])
