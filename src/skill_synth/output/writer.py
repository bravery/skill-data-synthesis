"""JSONL output writer for training samples."""

from __future__ import annotations

import json
from pathlib import Path

from ..config import OutputConfig
from ..models import TrainingSample


class JSONLWriter:
    """Writes training samples to JSONL format."""

    def __init__(self, config: OutputConfig) -> None:
        self._dir = Path(config.dir)
        self._filename = config.filename
        self._stats_filename = config.stats_filename

    def write(self, samples: list[TrainingSample], stats: dict | None = None) -> Path:
        """Write samples to JSONL file. Returns the output path."""
        self._dir.mkdir(parents=True, exist_ok=True)
        output_path = self._dir / self._filename

        with open(output_path, "w", encoding="utf-8") as f:
            for sample in samples:
                record = {
                    "instruction": sample.instruction,
                    "response": sample.response,
                    "system_prompt": sample.system_prompt,
                    "skill_id": sample.skill_id,
                    "strategy": sample.strategy.value,
                    "difficulty": sample.difficulty,
                    "quality_score": sample.quality_score,
                    "metadata": sample.metadata,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        # Write stats if provided
        if stats:
            stats_path = self._dir / self._stats_filename
            with open(stats_path, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)

        return output_path
