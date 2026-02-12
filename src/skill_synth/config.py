"""Configuration loading from YAML + environment variables."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.8
    max_tokens: int = 2048

    @property
    def api_key(self) -> str:
        key = os.getenv(self.api_key_env, "")
        if not key:
            raise ValueError(f"Environment variable {self.api_key_env} is not set")
        return key


class AwesomePromptsConfig(BaseModel):
    enabled: bool = True
    url: str = "https://raw.githubusercontent.com/f/awesome-chatgpt-prompts/main/prompts.csv"


class GitHubRepoConfig(BaseModel):
    repo: str
    paths: list[str] = Field(default_factory=lambda: [""])
    patterns: list[str] = Field(default_factory=lambda: ["*.md"])
    parser: str = "markdown"


class SourcesConfig(BaseModel):
    awesome_prompts: AwesomePromptsConfig = Field(default_factory=AwesomePromptsConfig)
    github_repos: list[GitHubRepoConfig] = Field(default_factory=list)


class SkillFilterConfig(BaseModel):
    min_length: int = 50
    max_length: int = 10000
    min_words: int = 10
    ascii_ratio: float = 0.7
    blacklist_patterns: list[str] = Field(default_factory=list)


class SampleFilterConfig(BaseModel):
    min_instruction_length: int = 10
    max_instruction_length: int = 2000
    min_response_length: int = 50
    max_response_length: int = 8000
    refusal_patterns: list[str] = Field(default_factory=list)


class LLMQualityConfig(BaseModel):
    enabled: bool = True
    sample_ratio: float = 0.2
    min_overall_score: float = 3.5


class FiltersConfig(BaseModel):
    skill: SkillFilterConfig = Field(default_factory=SkillFilterConfig)
    sample: SampleFilterConfig = Field(default_factory=SampleFilterConfig)
    llm_quality: LLMQualityConfig = Field(default_factory=LLMQualityConfig)


class SynthesisStrategyConfig(BaseModel):
    name: str
    samples_per_skill: int = 3


class SynthesisConfig(BaseModel):
    strategies: list[SynthesisStrategyConfig] = Field(default_factory=list)
    concurrency: int = 10
    temperature: float = 0.8


class OutputConfig(BaseModel):
    dir: str = "data/output"
    filename: str = "training_data.jsonl"
    stats_filename: str = "stats.json"


class AppConfig(BaseModel):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    sources: SourcesConfig = Field(default_factory=SourcesConfig)
    filters: FiltersConfig = Field(default_factory=FiltersConfig)
    synthesis: SynthesisConfig = Field(default_factory=SynthesisConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)


def load_config(config_path: str | Path = "config.yaml") -> AppConfig:
    """Load configuration from YAML file and environment variables."""
    load_dotenv()

    config_path = Path(config_path)
    data: dict[str, Any] = {}
    if config_path.exists():
        with open(config_path) as f:
            data = yaml.safe_load(f) or {}

    return AppConfig(**data)
