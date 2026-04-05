"""SKILL.md management implementing SkillStore protocol."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from evoagent.core.protocols import SkillStore

logger = logging.getLogger(__name__)


def parse_frontmatter(content: str) -> dict[str, Any]:
    """Parse YAML frontmatter from SKILL.md content."""
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", content, re.DOTALL)
    if not match:
        return {"name": "unknown", "description": "No description", "body": content}

    metadata: dict[str, str] = {}
    for line in match.group(1).split("\n"):
        if ":" in line:
            key, value = line.split(":", 1)
            metadata[key.strip()] = value.strip()

    return {
        "name": metadata.get("name", "unknown"),
        "description": metadata.get("description", "No description"),
        "body": match.group(2).strip(),
    }


class SkillManager(SkillStore):
    """File-backed SKILL.md CRUD with progressive disclosure."""

    def __init__(self, skills_dir: Path | str) -> None:
        self.skills_dir = Path(skills_dir)
        self.skills_dir.mkdir(parents=True, exist_ok=True)

    def discover(self) -> dict[str, dict[str, Any]]:
        skills: dict[str, dict[str, Any]] = {}
        for skill_dir in self.skills_dir.iterdir():
            if skill_dir.is_dir():
                skill_file = skill_dir / "SKILL.md"
                if skill_file.exists():
                    parsed = parse_frontmatter(skill_file.read_text())
                    skills[skill_dir.name] = {
                        "name": parsed["name"],
                        "description": parsed["description"],
                        "path": str(skill_file),
                    }
        return skills

    def load(self, name: str) -> str | None:
        skill_file = self.skills_dir / name / "SKILL.md"
        if not skill_file.exists():
            return None
        parsed = parse_frontmatter(skill_file.read_text())
        return parsed["body"]

    def create(self, name: str, description: str, content: str) -> Path:
        skill_id = name.lower().replace(" ", "-").replace("_", "-")
        skill_dir = self.skills_dir / skill_id
        skill_dir.mkdir(parents=True, exist_ok=True)
        skill_file = skill_dir / "SKILL.md"
        skill_file.write_text(f"---\nname: {name}\ndescription: {description}\n---\n\n{content}\n")
        logger.info("Created skill '%s' at %s", name, skill_file)
        return skill_file

    def validate(self, skill_path: Path) -> tuple[bool, str]:
        if not skill_path.exists():
            return False, f"File not found: {skill_path}"
        content = skill_path.read_text()
        if not content.startswith("---"):
            return False, "Missing YAML frontmatter"
        parsed = parse_frontmatter(content)
        if parsed["name"] == "unknown":
            return False, "Missing 'name' in frontmatter"
        if parsed["description"] == "No description":
            return False, "Missing 'description' in frontmatter"
        if not parsed["body"].strip():
            return False, "Skill body is empty"
        return True, "Valid"
