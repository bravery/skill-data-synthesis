"""Synthesizer: orchestrates LLM calls to generate training samples from skills."""

from __future__ import annotations

import asyncio

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn

from ..config import SynthesisConfig
from ..llm.base import BaseLLMClient
from ..models import Skill, SynthesisStrategy, TrainingSample
from .prompts import STRATEGY_PROMPTS

console = Console()


class Synthesizer:
    """Generates training samples from skills using multiple synthesis strategies."""

    def __init__(self, config: SynthesisConfig, llm: BaseLLMClient) -> None:
        self._cfg = config
        self._llm = llm
        self._sem = asyncio.Semaphore(config.concurrency)

    async def synthesize_all(self, skills: list[Skill]) -> list[TrainingSample]:
        """Generate training samples for all skills using all configured strategies."""
        all_samples: list[TrainingSample] = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            console=console,
        ) as progress:
            total_tasks = len(skills) * len(self._cfg.strategies)
            task = progress.add_task("Synthesizing...", total=total_tasks)

            coros = []
            for skill in skills:
                for strategy_cfg in self._cfg.strategies:
                    coros.append(
                        self._synthesize_one(skill, strategy_cfg.name, strategy_cfg.samples_per_skill)
                    )

            for coro in asyncio.as_completed(coros):
                samples = await coro
                all_samples.extend(samples)
                progress.advance(task)

        return all_samples

    async def _synthesize_one(
        self, skill: Skill, strategy_name: str, n_samples: int
    ) -> list[TrainingSample]:
        """Generate samples for a single skill with one strategy."""
        async with self._sem:
            prompt_template = STRATEGY_PROMPTS.get(strategy_name)
            if not prompt_template:
                return []

            prompt = prompt_template.format(
                skill_content=skill.content[:3000],  # truncate very long skills
                n=n_samples,
            )

            try:
                result = await self._llm.generate_json(
                    [
                        {
                            "role": "system",
                            "content": "You are a training data generation expert. Always respond with valid JSON.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self._cfg.temperature,
                    max_tokens=2048,
                )

                return self._parse_result(result, skill, strategy_name)
            except Exception as e:
                console.print(
                    f"  [yellow]⚠[/yellow] Synthesis failed for '{skill.name}' "
                    f"({strategy_name}): {e}"
                )
                return []

    def _parse_result(
        self, result: dict | list, skill: Skill, strategy_name: str
    ) -> list[TrainingSample]:
        """Parse LLM JSON response into TrainingSample objects."""
        strategy = SynthesisStrategy(strategy_name)

        # Handle both {"samples": [...]} and direct list
        if isinstance(result, dict):
            items = result.get("samples", [])
        elif isinstance(result, list):
            items = result
        else:
            return []

        samples: list[TrainingSample] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            instruction = item.get("instruction", "").strip()
            response = item.get("response", "").strip()
            if not instruction or not response:
                continue

            samples.append(
                TrainingSample(
                    instruction=instruction,
                    response=response,
                    skill_id=skill.id,
                    strategy=strategy,
                    difficulty=item.get("difficulty", "medium"),
                    metadata={
                        "skill_name": skill.name,
                        "edge_type": item.get("edge_type", ""),
                        "scenario": item.get("scenario", ""),
                    },
                )
            )

        return samples
