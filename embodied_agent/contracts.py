"""Stable contracts shared by the SO-101 agent layers.

This package is an integration scaffold. It deliberately has no hardware or
model-provider imports so adapters can be added without coupling the runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping


@dataclass(frozen=True, slots=True)
class Observation:
    run_id: str
    episode_id: str
    observed_at_ns: int
    task_id: str
    embodiment_id: str
    calibration_id: str
    joint_names: tuple[str, ...]
    joint_positions: tuple[float, ...]
    camera_refs: Mapping[str, str] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ActionChunk:
    policy_id: str
    joint_names: tuple[str, ...]
    steps: tuple[tuple[float, ...], ...]
    based_on_observation_ns: int
    valid_for_ms: int
    action_schema: str


@dataclass(frozen=True, slots=True)
class SkillCall:
    skill_id: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    rationale: str = ""
    request_human_approval: bool = False


@dataclass(frozen=True, slots=True)
class SkillManifest:
    skill_id: str
    version: str
    embodiments: tuple[str, ...]
    action_schema: str
    preconditions: tuple[str, ...]
    postconditions: tuple[str, ...]
    max_duration_s: float
    max_retries: int = 0
    requires_human_approval: bool = False
    verification_status: Literal["experimental", "verified", "blocked"] = "experimental"


@dataclass(frozen=True, slots=True)
class DemoReference:
    episode_id: str
    task_id: str
    embodiment_id: str
    video_uri: str
    action_data_uri: str | None
    outcome: Literal["success", "failure", "unknown"]
    tags: tuple[str, ...] = ()
    calibration_id: str | None = None


@dataclass(frozen=True, slots=True)
class SkillResult:
    status: Literal["proposed", "completed", "retryable", "needs_human", "failed"]
    action: ActionChunk | None = None
    evidence: Mapping[str, Any] = field(default_factory=dict)
    failure_code: str | None = None


@dataclass(frozen=True, slots=True)
class EpisodeManifest:
    episode_id: str
    run_id: str
    task_id: str
    embodiment_id: str
    calibration_id: str
    started_at: str
    media: Mapping[str, str]
    action_data_uri: str | None
    model_versions: Mapping[str, str]
    skill_versions: Mapping[str, str]
    seed: int | None = None
    outcome: Literal["success", "failure", "interrupted", "unknown"] = "unknown"
    failure_code: str | None = None
    human_interventions: tuple[Mapping[str, Any], ...] = ()

