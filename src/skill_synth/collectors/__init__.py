"""Collector factory and aggregation."""

from __future__ import annotations

import asyncio

from rich.console import Console

from ..config import SourcesConfig
from ..models import Skill
from .awesome_prompts import AwesomePromptsCollector
from .base import BaseCollector
from .github_repo import GitHubRepoCollector

console = Console()


def _build_collectors(sources: SourcesConfig) -> list[BaseCollector]:
    """Build collector instances from config."""
    collectors: list[BaseCollector] = []

    if sources.awesome_prompts.enabled:
        collectors.append(AwesomePromptsCollector(url=sources.awesome_prompts.url))

    for repo_cfg in sources.github_repos:
        collectors.append(GitHubRepoCollector(config=repo_cfg))

    return collectors


async def collect_all(sources: SourcesConfig) -> list[Skill]:
    """Run all collectors concurrently and deduplicate by content hash."""
    collectors = _build_collectors(sources)

    async def _run(c: BaseCollector) -> list[Skill]:
        try:
            skills = await c.collect()
            console.print(f"  [green]✓[/green] {c.name}: {len(skills)} skills")
            return skills
        except Exception as e:
            console.print(f"  [red]✗[/red] {c.name}: {e}")
            return []

    results = await asyncio.gather(*[_run(c) for c in collectors])
    all_skills = [s for batch in results for s in batch]

    # Deduplicate by content hash
    seen: set[str] = set()
    unique: list[Skill] = []
    for skill in all_skills:
        if skill.id not in seen:
            seen.add(skill.id)
            unique.append(skill)

    return unique


__all__ = ["collect_all"]
