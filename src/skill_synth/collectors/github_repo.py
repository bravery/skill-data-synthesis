"""GitHub repository collector using the Trees API."""

from __future__ import annotations

import fnmatch
import json
import os
import re
from pathlib import PurePosixPath

import httpx

from ..config import GitHubRepoConfig
from ..models import Skill
from .base import BaseCollector


class GitHubRepoCollector(BaseCollector):
    """Collects skill/prompt files from a GitHub repository.

    Uses the Git Trees API for efficient directory listing, then fetches
    matching files via raw.githubusercontent.com.
    """

    def __init__(self, config: GitHubRepoConfig) -> None:
        self._config = config
        self._repo = config.repo

    @property
    def name(self) -> str:
        return f"github:{self._repo}"

    async def collect(self) -> list[Skill]:
        token = os.getenv("GITHUB_TOKEN", "")
        headers: dict[str, str] = {"Accept": "application/vnd.github.v3+json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"

        async with httpx.AsyncClient(timeout=30, headers=headers) as client:
            # Get full repo tree
            tree_url = f"https://api.github.com/repos/{self._repo}/git/trees/main?recursive=1"
            resp = await client.get(tree_url)
            if resp.status_code == 404:
                # Try 'master' branch
                tree_url = f"https://api.github.com/repos/{self._repo}/git/trees/master?recursive=1"
                resp = await client.get(tree_url)
            resp.raise_for_status()

            tree = resp.json().get("tree", [])
            matching_paths = self._filter_paths(tree)

            # Fetch file contents
            skills: list[Skill] = []
            for path in matching_paths:
                raw_url = f"https://raw.githubusercontent.com/{self._repo}/main/{path}"
                try:
                    file_resp = await client.get(raw_url)
                    if file_resp.status_code == 404:
                        raw_url = f"https://raw.githubusercontent.com/{self._repo}/master/{path}"
                        file_resp = await client.get(raw_url)
                    file_resp.raise_for_status()
                except httpx.HTTPError:
                    continue

                content = file_resp.text.strip()
                if not content:
                    continue

                parsed = self._parse_content(content, path)
                if parsed:
                    skills.append(parsed)

        return skills

    def _filter_paths(self, tree: list[dict]) -> list[str]:
        """Filter tree entries by configured paths and patterns."""
        result = []
        for entry in tree:
            if entry.get("type") != "blob":
                continue
            path = entry["path"]
            # Check if file is under any of the configured paths
            path_match = not self._config.paths or any(
                path.startswith(p) for p in self._config.paths
            )
            if not path_match:
                continue
            # Check file pattern match
            filename = PurePosixPath(path).name
            pattern_match = any(
                fnmatch.fnmatch(filename, pat) for pat in self._config.patterns
            )
            if pattern_match:
                result.append(path)
        return result

    def _parse_content(self, content: str, path: str) -> Skill | None:
        """Parse file content into a Skill based on the configured parser."""
        parser = self._config.parser

        if parser == "markdown":
            return self._parse_markdown(content, path)
        elif parser == "json":
            return self._parse_json(content, path)
        else:
            return self._parse_plaintext(content, path)

    def _parse_markdown(self, content: str, path: str) -> Skill | None:
        """Extract skill content from a markdown file."""
        # Try to extract title from first heading
        title_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
        title = title_match.group(1).strip() if title_match else PurePosixPath(path).stem

        return Skill(
            name=title,
            content=content,
            source=f"github:{self._repo}/{path}",
            tags=["github", "markdown"],
            metadata={"repo": self._repo, "path": path},
        )

    def _parse_json(self, content: str, path: str) -> Skill | None:
        """Extract skill from a JSON file with a 'System Prompt' field."""
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            return None
        if not isinstance(data, dict):
            return None

        # Try common field names for the prompt content
        prompt = (
            data.get("System Prompt")
            or data.get("system_prompt")
            or data.get("prompt")
            or ""
        )
        if not isinstance(prompt, str) or not prompt.strip():
            return None

        name = (
            data.get("agent_name")
            or data.get("name")
            or data.get("title")
            or PurePosixPath(path).stem
        )

        return Skill(
            name=name,
            content=prompt.strip(),
            source=f"github:{self._repo}/{path}",
            tags=["github", "json"],
            metadata={"repo": self._repo, "path": path},
        )

    def _parse_plaintext(self, content: str, path: str) -> Skill | None:
        """Create a skill from plain text content."""
        title = PurePosixPath(path).stem.replace("-", " ").replace("_", " ").title()
        return Skill(
            name=title,
            content=content,
            source=f"github:{self._repo}/{path}",
            tags=["github", "plaintext"],
            metadata={"repo": self._repo, "path": path},
        )
