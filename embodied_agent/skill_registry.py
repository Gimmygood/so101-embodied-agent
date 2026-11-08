"""Versioned skill allow-list for the live runtime."""

from __future__ import annotations

from embodied_agent.contracts import SkillManifest
from embodied_agent.interfaces import Skill


class SkillRegistry:
    """Only verified skill versions are returned to the high-level brain."""

    def __init__(self) -> None:
        self._skills: dict[tuple[str, str], Skill] = {}
        self._active_versions: dict[str, str] = {}

    def register(self, skill: Skill) -> None:
        manifest = skill.manifest
        if manifest.verification_status != "verified":
            raise ValueError("Only verified skills can enter the live registry")
        if not manifest.skill_id or not manifest.version:
            raise ValueError("Skill ID and version are required")
        if manifest.max_duration_s <= 0 or manifest.max_retries < 0:
            raise ValueError("Skill time budget and retry count are invalid")
        key = (manifest.skill_id, manifest.version)
        if key in self._skills:
            raise ValueError(f"Skill version already registered: {key}")
        self._skills[key] = skill

    def available(self, embodiment_id: str) -> tuple[SkillManifest, ...]:
        return tuple(
            skill.manifest
            for (skill_id, version), skill in self._skills.items()
            if self._active_versions.get(skill_id) == version
            and embodiment_id in skill.manifest.embodiments
        )

    def activate(self, skill_id: str, version: str) -> None:
        """Set a reviewed active version; callers own the human approval step."""
        skill = self._skills.get((skill_id, version))
        if skill is None or skill.manifest.verification_status != "verified":
            raise LookupError("Only registered, verified skill versions can activate")
        self._active_versions[skill_id] = version

    def resolve(self, skill_id: str, embodiment_id: str) -> Skill:
        active_version = self._active_versions.get(skill_id)
        skill = self._skills.get((skill_id, active_version)) if active_version else None
        if skill is None or embodiment_id not in skill.manifest.embodiments:
            raise LookupError(
                f"No verified skill {skill_id!r} for {embodiment_id!r}"
            )
        return skill

