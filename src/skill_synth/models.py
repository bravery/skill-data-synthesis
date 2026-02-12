"""Pydantic data models for the synthesis pipeline."""

from __future__ import annotations

from enum import Enum
from typing import Any

import xxhash
from pydantic import BaseModel, Field, computed_field


class SynthesisStrategy(str, Enum):
    DIRECT_TASK = "direct_task"
    SCENARIO_BASED = "scenario_based"
    CONVERSATION = "conversation"
    EDGE_CASE = "edge_case"


class Skill(BaseModel):
    """A collected skill/prompt with metadata."""

    name: str
    content: str
    source: str
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @computed_field
    @property
    def id(self) -> str:
        return xxhash.xxh64(self.content.strip().lower()).hexdigest()


class TrainingSample(BaseModel):
    """A single training example produced by synthesis."""

    instruction: str
    response: str
    system_prompt: str = ""
    skill_id: str
    strategy: SynthesisStrategy
    difficulty: str = "medium"
    quality_score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class QualityVerdict(BaseModel):
    """LLM quality assessment for a training sample."""

    clarity: float = Field(ge=1, le=5)
    accuracy: float = Field(ge=1, le=5)
    helpfulness: float = Field(ge=1, le=5)
    alignment: float = Field(ge=1, le=5)
    value: float = Field(ge=1, le=5)
    reasoning: str = ""

    @computed_field
    @property
    def overall(self) -> float:
        return round(
            (self.clarity + self.accuracy + self.helpfulness + self.alignment + self.value) / 5,
            2,
        )
