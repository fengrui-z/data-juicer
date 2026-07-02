"""Static batch recommendations derived from the existing probe mechanism."""

import json
from pathlib import Path
from typing import Iterable, List, Optional

from ..contracts import BatchRecommendation


class StaticBatchRecommender:
    """Build and optionally apply a deterministic per-operator batch plan."""

    def __init__(self, output_dir: Optional[str] = None):
        self.output_path: Optional[Path] = None
        if output_dir:
            directory = Path(output_dir)
            directory.mkdir(parents=True, exist_ok=True)
            self.output_path = directory / "batch_recommendations.json"

    def recommend(
        self,
        operators: Iterable,
        candidate_batch_sizes: Iterable[int],
    ) -> List[BatchRecommendation]:
        operators = list(operators)
        candidates = list(candidate_batch_sizes)
        if len(operators) != len(candidates):
            raise ValueError("candidate batch sizes must match operators")

        recommendations = []
        for operator, candidate in zip(operators, candidates):
            is_batched = bool(callable(getattr(operator, "is_batched_op", None)) and operator.is_batched_op())
            current = max(1, int(getattr(operator, "batch_size", 1)))
            recommended = max(1, int(candidate)) if is_batched else current
            recommendations.append(
                BatchRecommendation(
                    stage_name=getattr(
                        operator,
                        "_name",
                        operator.__class__.__name__,
                    ),
                    current_batch_size=current,
                    recommended_batch_size=recommended,
                    eligible=is_batched,
                    reason=("resource_probe" if is_batched else "operator_does_not_support_batching"),
                )
            )

        self._persist(recommendations)
        return recommendations

    @staticmethod
    def apply(operators: Iterable, recommendations: Iterable[BatchRecommendation]):
        operators = list(operators)
        recommendations = list(recommendations)
        if len(operators) != len(recommendations):
            raise ValueError("recommendations must match operators")

        for operator, recommendation in zip(operators, recommendations):
            if recommendation.eligible:
                operator.batch_size = recommendation.recommended_batch_size

    def _persist(self, recommendations: List[BatchRecommendation]):
        if self.output_path is None:
            return
        temporary_path = self.output_path.with_suffix(".tmp")
        with temporary_path.open("w", encoding="utf-8") as output:
            json.dump(
                [recommendation.to_dict() for recommendation in recommendations],
                output,
                indent=2,
                sort_keys=True,
            )
            output.write("\n")
        temporary_path.replace(self.output_path)
