"""Abstract collector interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import Skill


class BaseCollector(ABC):
    """Base class for skill/prompt collectors."""

    @abstractmethod
    async def collect(self) -> list[Skill]:
        """Collect skills from the source. Returns a list of Skill objects."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this collector."""
