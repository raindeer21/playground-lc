from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable
import logging

_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)


@dataclass(slots=True)
class Skill:
    skill_id: str
    name: str
    description: str
    body: str
    path: Path
    tool_whitelist: list[str]


class SkillStore:
    def __init__(self, root: str | Path = "skills") -> None:
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging.DEBUG)
        self.root = Path(root)
        self._skills = self._load_skills()

    def _load_skills(self) -> dict[str, Skill]:
        skills: dict[str, Skill] = {}
        if not self.root.exists():
            self._logger.error(f"ROOT NOT FOUND: {self.root.absolute()}")
            return skills

        for skill_md in self.root.glob("*/SKILL.md"):
            text = skill_md.read_text(encoding="utf-8")
            frontmatter = _extract_frontmatter(text)
            skill_id = skill_md.parent.name
            skills[skill_id] = Skill(
                skill_id=skill_id,
                name=frontmatter.get("name", skill_id),
                description=frontmatter.get("description", ""),
                body=text,
                path=skill_md,
                tool_whitelist=_parse_allowed_tools(frontmatter),
            )

        return skills

    def headers(self) -> list[dict[str, str]]:
        return [
            {
                "skill_id": skill.skill_id,
                "description": skill.description,
            }
            for skill in self._skills.values()
        ]

    def get(self, skill_id: str) -> Skill | None:
        return self._skills.get(skill_id)

    def all(self) -> Iterable[Skill]:
        return self._skills.values()

    def tool_whitelist_for(self, skill_ids: list[str]) -> set[str] | None:
        if skill_ids is None:
            return None

        tool_names: set[str] = set()
        for skill_id in skill_ids:
            skill = self._skills.get(skill_id)
            if not skill:
                continue
            tool_names.update(skill.tool_whitelist)

        return tool_names


def _extract_frontmatter(text: str) -> dict[str, str]:
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return {}

    frontmatter: dict[str, str] = {}
    for raw_line in match.group(1).splitlines():
        if ":" not in raw_line:
            continue
        key, value = raw_line.split(":", 1)
        frontmatter[key.strip()] = value.strip()
    return frontmatter


def _parse_allowed_tools(frontmatter: dict[str, str]) -> list[str]:
    raw = frontmatter.get("allowed-tools") or frontmatter.get("allowed_tools")
    if not raw:
        return []

    normalized = []
    for item in re.split(r"[,\n]", raw):
        name = item.strip().strip("[]\"'")
        if not name:
            continue
        normalized.append(_normalize_tool_name(name))
    return normalized


def _normalize_tool_name(name: str) -> str:
    converted = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    return converted.replace("-", "_").lower()
