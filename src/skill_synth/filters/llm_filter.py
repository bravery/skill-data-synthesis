"""LLM-based quality filter for training samples."""

from __future__ import annotations

import asyncio
import random

from rich.console import Console

from ..config import LLMQualityConfig
from ..llm.base import BaseLLMClient
from ..models import QualityVerdict, TrainingSample

console = Console()

QUALITY_PROMPT = """\
You are a training data quality evaluator. Assess the following instruction-response pair on these criteria (1-5 scale each):

1. **Clarity**: Is the instruction clear and well-formed?
2. **Accuracy**: Is the response factually correct and precise?
3. **Helpfulness**: Does the response fully address the instruction?
4. **Alignment**: Is the response safe, ethical, and well-aligned?
5. **Value**: Does this example teach useful knowledge or skills?

Instruction:
{instruction}

Response:
{response}

Respond with ONLY a JSON object:
{{"clarity": <1-5>, "accuracy": <1-5>, "helpfulness": <1-5>, "alignment": <1-5>, "value": <1-5>, "reasoning": "<brief explanation>"}}"""


class LLMQualityFilter:
    """Scores a sample of training data using an LLM judge."""

    def __init__(self, config: LLMQualityConfig, llm: BaseLLMClient) -> None:
        self._cfg = config
        self._llm = llm

    async def filter(
        self, samples: list[TrainingSample]
    ) -> tuple[list[TrainingSample], dict]:
        """Score a random subset and return all samples with quality metrics.

        Returns (filtered_samples, quality_stats).
        """
        if not self._cfg.enabled or not samples:
            return samples, {}

        # Sample a subset for evaluation
        sample_count = max(1, int(len(samples) * self._cfg.sample_ratio))
        to_evaluate = random.sample(samples, min(sample_count, len(samples)))

        console.print(f"  Evaluating {len(to_evaluate)} samples with LLM...")

        sem = asyncio.Semaphore(5)
        verdicts: list[tuple[TrainingSample, QualityVerdict | None]] = []

        async def _evaluate(s: TrainingSample) -> tuple[TrainingSample, QualityVerdict | None]:
            async with sem:
                try:
                    result = await self._llm.generate_json(
                        [
                            {
                                "role": "user",
                                "content": QUALITY_PROMPT.format(
                                    instruction=s.instruction, response=s.response
                                ),
                            }
                        ],
                        temperature=0.1,
                        max_tokens=512,
                    )
                    verdict = QualityVerdict(**result)  # type: ignore[arg-type]
                    return s, verdict
                except Exception:
                    return s, None

        results = await asyncio.gather(*[_evaluate(s) for s in to_evaluate])
        verdicts = list(results)

        # Compute stats
        scored = [(s, v) for s, v in verdicts if v is not None]
        if not scored:
            return samples, {"evaluated": 0}

        scores = [v.overall for _, v in scored]
        passed = sum(1 for sc in scores if sc >= self._cfg.min_overall_score)
        stats = {
            "evaluated": len(scored),
            "mean_score": round(sum(scores) / len(scores), 2),
            "min_score": min(scores),
            "max_score": max(scores),
            "pass_rate": round(passed / len(scored), 2),
        }

        # Mark quality scores on evaluated samples
        evaluated_ids = set()
        for sample, verdict in scored:
            if verdict:
                sample.quality_score = verdict.overall
                evaluated_ids.add(id(sample))

        # Keep all samples (quality check is for metrics, not hard filtering)
        # But remove clearly bad samples that were evaluated and failed
        fail_ids: set[int] = set()
        for sample, verdict in scored:
            if verdict and verdict.overall < self._cfg.min_overall_score:
                fail_ids.add(id(sample))

        filtered = [s for s in samples if id(s) not in fail_ids]
        return filtered, stats
