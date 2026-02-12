"""Pipeline orchestrator - coordinates all stages of the synthesis pipeline."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .collectors import collect_all
from .config import AppConfig
from .filters import HeuristicFilter, LLMQualityFilter, SampleHeuristicFilter
from .llm import create_llm_client
from .output import JSONLWriter
from .synthesis import Synthesizer

console = Console()


class Pipeline:
    """Main pipeline orchestrator.

    Stages:
    1. Collect skills from configured sources
    2. Filter skills with heuristic rules
    3. Synthesize training samples via LLM
    4. Filter samples with heuristic rules
    5. LLM quality check (sample-based)
    6. Write JSONL output
    """

    def __init__(self, config: AppConfig) -> None:
        self._cfg = config

    async def run(self, *, dry_run: bool = False, limit: int | None = None) -> None:
        """Execute the full pipeline."""
        console.print(Panel("[bold]Skill Data Synthesis Pipeline[/bold]", style="blue"))

        stats: dict = {}

        # Stage 1: Collect
        console.print("\n[bold cyan]Stage 1/6: Collecting skills[/bold cyan]")
        skills = await collect_all(self._cfg.sources)
        stats["raw_skills"] = len(skills)
        console.print(f"  Collected {len(skills)} unique skills")

        if limit:
            skills = skills[:limit]
            console.print(f"  Limited to {len(skills)} skills")

        # Stage 2: Filter skills
        console.print("\n[bold cyan]Stage 2/6: Filtering skills[/bold cyan]")
        skill_filter = HeuristicFilter(self._cfg.filters.skill)
        skills = skill_filter.filter(skills)
        stats["filtered_skills"] = len(skills)
        console.print(f"  {len(skills)} skills passed heuristic filter")

        if dry_run:
            console.print("\n[yellow]Dry run mode — stopping after collection and filtering.[/yellow]")
            self._print_summary(stats, skills_preview=skills[:5])
            return

        # Stage 3: Synthesize
        console.print("\n[bold cyan]Stage 3/6: Synthesizing training samples[/bold cyan]")
        llm = create_llm_client(self._cfg.llm)
        try:
            synthesizer = Synthesizer(self._cfg.synthesis, llm)
            samples = await synthesizer.synthesize_all(skills)
            stats["raw_samples"] = len(samples)
            console.print(f"  Generated {len(samples)} raw samples")

            # Stage 4: Filter samples
            console.print("\n[bold cyan]Stage 4/6: Filtering samples[/bold cyan]")
            sample_filter = SampleHeuristicFilter(self._cfg.filters.sample)
            samples = sample_filter.filter(samples)
            stats["filtered_samples"] = len(samples)
            console.print(f"  {len(samples)} samples passed heuristic filter")

            # Stage 5: LLM quality check
            console.print("\n[bold cyan]Stage 5/6: LLM quality evaluation[/bold cyan]")
            quality_filter = LLMQualityFilter(self._cfg.filters.llm_quality, llm)
            samples, quality_stats = await quality_filter.filter(samples)
            stats["quality"] = quality_stats
            stats["final_samples"] = len(samples)
            console.print(f"  {len(samples)} samples after quality filtering")
            if quality_stats:
                console.print(
                    f"  Quality: mean={quality_stats.get('mean_score', 'N/A')}, "
                    f"pass_rate={quality_stats.get('pass_rate', 'N/A')}"
                )
        finally:
            await llm.close()

        # Stage 6: Write output
        console.print("\n[bold cyan]Stage 6/6: Writing output[/bold cyan]")
        writer = JSONLWriter(self._cfg.output)
        output_path = writer.write(samples, stats)
        console.print(f"  Written to [green]{output_path}[/green]")

        self._print_summary(stats)

    def _print_summary(self, stats: dict, skills_preview: list | None = None) -> None:
        """Print a summary table of pipeline results."""
        table = Table(title="Pipeline Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        for key, value in stats.items():
            if isinstance(value, dict):
                for k, v in value.items():
                    table.add_row(f"  {key}.{k}", str(v))
            else:
                table.add_row(key, str(value))

        console.print(table)

        if skills_preview:
            console.print("\n[bold]Sample skills:[/bold]")
            for skill in skills_preview:
                console.print(f"  • {skill.name} ({skill.source})")
                console.print(f"    {skill.content[:100]}...")

        # Strategy breakdown
        if "final_samples" in stats:
            console.print(f"\n[bold green]Done![/bold green] {stats['final_samples']} training samples generated.")
