"""Minimal skill adapter that asks one routed VLA for one bounded chunk."""

from __future__ import annotations

from embodied_agent.contracts import Observation, SkillCall, SkillManifest, SkillResult
from embodied_agent.interfaces import VLAAdapter


class PolicyChunkSkill:
    """A reusable leaf skill; task-specific sequencing belongs in a Skill."""

    def __init__(self, manifest: SkillManifest) -> None:
        self._manifest = manifest

    @property
    def manifest(self) -> SkillManifest:
        return self._manifest

    def propose(
        self,
        call: SkillCall,
        observation: Observation,
        policy: VLAAdapter,
    ) -> SkillResult:
        if self._manifest.verification_status != "verified":
            return SkillResult(status="failed", failure_code="skill_not_verified")
        action = policy.predict(observation, skill_id=self._manifest.skill_id)
        return SkillResult(status="proposed", action=action)

