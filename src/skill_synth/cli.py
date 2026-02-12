"""CLI entry point for the skill data synthesis pipeline."""

from __future__ import annotations

import argparse
import asyncio
import sys

from rich.console import Console

from .config import load_config
from .pipeline import Pipeline

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="skill-synth",
        description="Automated pipeline for synthesizing LLM training data from skill/prompt collections",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config YAML file (default: config.yaml)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only collect and filter skills, skip synthesis",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of skills to process",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        config = load_config(args.config)
    except Exception as e:
        console.print(f"[red]Failed to load config:[/red] {e}")
        sys.exit(1)

    pipeline = Pipeline(config)

    try:
        asyncio.run(pipeline.run(dry_run=args.dry_run, limit=args.limit))
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"[red]Pipeline failed:[/red] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
