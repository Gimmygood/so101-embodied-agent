"""Pluggable interfaces for brain, memory, skills, policies and robot I/O."""

from __future__ import annotations

from typing import Protocol, Sequence

from embodied_agent.contracts import (
    ActionChunk,
    DemoReference,
    Observation,
    SkillCall,
    SkillManifest,
    SkillResult,
)


class Brain(Protocol):
    """High-level planner; implement this with GPT-6 Astra or another provider."""

    def choose_skill(
        self,
        goal: str,
        observation: Observation,
        demonstrations: Sequence[DemoReference],
        allowed_skills: Sequence[SkillManifest],
    ) -> SkillCall: ...


class DemoMemory(Protocol):
    """Retrieves approved demonstrations for in-context learning."""

    def retrieve(
        self, goal: str, observation: Observation, *, limit: int
    ) -> Sequence[DemoReference]: ...


class VLAAdapter(Protocol):
    """One embodiment-compatible action policy behind the common interface."""

    model_id: str

    def is_ready(self) -> bool: ...

    def predict(self, observation: Observation, *, skill_id: str) -> ActionChunk: ...


class Skill(Protocol):
    """A versioned, bounded behavior that may invoke a routed VLA policy.

    ``propose`` is inference-only: it must return a proposed chunk and must
    never call the physical RobotBackend. AgentRuntime runs the safety check
    before the backend is allowed to execute that chunk.
    """

    @property
    def manifest(self) -> SkillManifest: ...

    def propose(
        self,
        call: SkillCall,
        observation: Observation,
        policy: VLAAdapter,
    ) -> SkillResult: ...


class RobotBackend(Protocol):
    """SO-101 or simulator adapter. Hardware stop must be independent of the LLM."""

    def health(self) -> tuple[bool, str]: ...

    def observe(self) -> Observation: ...

    def execute(self, action: ActionChunk) -> Mapping[str, object]: ...

    def safe_stop(self, reason: str) -> None: ...


class HumanApproval(Protocol):
    def request(self, reason: str, observation: Observation) -> bool: ...

