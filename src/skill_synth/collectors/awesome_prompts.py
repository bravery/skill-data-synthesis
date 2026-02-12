"""Collector for awesome-chatgpt-prompts CSV."""

from __future__ import annotations

import csv
import io

import httpx

from ..models import Skill
from .base import BaseCollector


class AwesomePromptsCollector(BaseCollector):
    """Fetches prompts from the awesome-chatgpt-prompts CSV file."""

    def __init__(self, url: str) -> None:
        self._url = url

    @property
    def name(self) -> str:
        return "awesome-chatgpt-prompts"

    async def collect(self) -> list[Skill]:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(self._url)
            resp.raise_for_status()

        skills: list[Skill] = []
        reader = csv.DictReader(io.StringIO(resp.text))
        for row in reader:
            act = row.get("act", "").strip()
            prompt = row.get("prompt", "").strip()
            if not prompt:
                continue
            skills.append(
                Skill(
                    name=act or "untitled",
                    content=prompt,
                    source="awesome-chatgpt-prompts",
                    tags=["csv", "awesome-prompts"],
                )
            )
        return skills
