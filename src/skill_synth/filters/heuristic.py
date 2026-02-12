"""Heuristic filters for skills and training samples."""

from __future__ import annotations

import re

from ..config import SampleFilterConfig, SkillFilterConfig
from ..models import Skill, TrainingSample


class HeuristicFilter:
    """Filter skills by length, language, and blacklist patterns."""

    def __init__(self, config: SkillFilterConfig) -> None:
        self._cfg = config
        self._blacklist_re = [
            re.compile(re.escape(p), re.IGNORECASE) for p in config.blacklist_patterns
        ]

    def filter(self, skills: list[Skill]) -> list[Skill]:
        return [s for s in skills if self._accept(s)]

    def _accept(self, skill: Skill) -> bool:
        content = skill.content.strip()

        # Length check
        if len(content) < self._cfg.min_length or len(content) > self._cfg.max_length:
            return False

        # Word count check
        if len(content.split()) < self._cfg.min_words:
            return False

        # ASCII ratio check (language filter)
        if content:
            ascii_count = sum(1 for c in content if ord(c) < 128)
            if ascii_count / len(content) < self._cfg.ascii_ratio:
                return False

        # Blacklist patterns
        for pattern in self._blacklist_re:
            if pattern.search(content):
                return False

        return True


class SampleHeuristicFilter:
    """Filter training samples by length, duplication, and refusal patterns."""

    def __init__(self, config: SampleFilterConfig) -> None:
        self._cfg = config
        self._refusal_re = [
            re.compile(re.escape(p), re.IGNORECASE) for p in config.refusal_patterns
        ]

    def filter(self, samples: list[TrainingSample]) -> list[TrainingSample]:
        seen_instructions: set[str] = set()
        result: list[TrainingSample] = []

        for sample in samples:
            if not self._accept(sample):
                continue
            # Dedup by normalized instruction
            key = sample.instruction.strip().lower()
            if key in seen_instructions:
                continue
            seen_instructions.add(key)
            result.append(sample)

        return result

    def _accept(self, sample: TrainingSample) -> bool:
        inst = sample.instruction.strip()
        resp = sample.response.strip()

        # Instruction length
        if len(inst) < self._cfg.min_instruction_length:
            return False
        if len(inst) > self._cfg.max_instruction_length:
            return False

        # Response length
        if len(resp) < self._cfg.min_response_length:
            return False
        if len(resp) > self._cfg.max_response_length:
            return False

        # Refusal pattern detection (check start of response)
        response_start = resp[:200]
        for pattern in self._refusal_re:
            if pattern.search(response_start):
                return False

        return True
