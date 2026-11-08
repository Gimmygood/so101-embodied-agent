"""Fail-closed action validation before commands reach a robot backend."""

from __future__ import annotations

from dataclasses import dataclass
import math

from embodied_agent.contracts import ActionChunk, Observation


@dataclass(frozen=True, slots=True)
class JointLimit:
    minimum: float
    maximum: float
    max_delta_per_step: float


@dataclass(frozen=True, slots=True)
class SafetyProfile:
    embodiment_id: str
    action_schema: str
    joint_names: tuple[str, ...]
    joint_limits: tuple[JointLimit, ...]
    max_chunk_steps: int = 1
    max_observation_age_ms: int = 200
    max_action_validity_ms: int = 300


@dataclass(frozen=True, slots=True)
class SafetyDecision:
    allowed: bool
    code: str
    detail: str = ""


class SafetySupervisor:
    """Validates timestamps, schema, finite values and calibrated joint bounds.

    Limits must come from the actual SO-101 calibration/profile. Invalid commands
    are rejected, never silently clipped. A rejected command should trigger the
    backend's independent safe_stop and a human-visible event.
    """

    def __init__(self, profile: SafetyProfile) -> None:
        if not profile.joint_names or len(profile.joint_names) != len(profile.joint_limits):
            raise ValueError("Safety profile needs one calibrated limit per joint")
        if profile.max_chunk_steps < 1:
            raise ValueError("max_chunk_steps must be positive")
        if profile.max_observation_age_ms <= 0 or profile.max_action_validity_ms <= 0:
            raise ValueError("Observation and action validity limits must be positive")
        if any(
            limit.minimum >= limit.maximum or limit.max_delta_per_step <= 0
            for limit in profile.joint_limits
        ):
            raise ValueError("Joint limits and per-step deltas must be calibrated and positive")
        self.profile = profile

    def validate(
        self, observation: Observation, action: ActionChunk, *, now_ns: int
    ) -> SafetyDecision:
        profile = self.profile
        if observation.embodiment_id != profile.embodiment_id:
            return SafetyDecision(False, "wrong_embodiment")
        if action.action_schema != profile.action_schema:
            return SafetyDecision(False, "wrong_action_schema")
        if action.joint_names != profile.joint_names:
            return SafetyDecision(False, "joint_order_mismatch")
        if observation.joint_names != profile.joint_names:
            return SafetyDecision(False, "observation_joint_order_mismatch")
        if len(observation.joint_positions) != len(profile.joint_names):
            return SafetyDecision(False, "observation_shape_mismatch")
        if any(not math.isfinite(value) for value in observation.joint_positions):
            return SafetyDecision(False, "invalid_joint_state")
        max_age_ns = profile.max_observation_age_ms * 1_000_000
        age_ns = now_ns - observation.observed_at_ns
        if age_ns < 0 or age_ns > max_age_ns:
            return SafetyDecision(False, "stale_observation")
        if action.based_on_observation_ns != observation.observed_at_ns:
            return SafetyDecision(False, "observation_action_mismatch")
        if action.valid_for_ms <= 0 or action.valid_for_ms > profile.max_action_validity_ms:
            return SafetyDecision(False, "invalid_action_expiry")
        if not action.steps or len(action.steps) > profile.max_chunk_steps:
            return SafetyDecision(False, "invalid_chunk_length")
        for step_index, step in enumerate(action.steps):
            if len(step) != len(profile.joint_names):
                return SafetyDecision(False, "action_shape_mismatch", str(step_index))
            for joint_index, value in enumerate(step):
                if not math.isfinite(value):
                    return SafetyDecision(False, "non_finite_action", str(step_index))
                limit = profile.joint_limits[joint_index]
                if value < limit.minimum or value > limit.maximum:
                    return SafetyDecision(False, "joint_limit", profile.joint_names[joint_index])
                previous = (
                    observation.joint_positions[joint_index]
                    if step_index == 0
                    else action.steps[step_index - 1][joint_index]
                )
                if abs(value - previous) > limit.max_delta_per_step:
                    return SafetyDecision(False, "joint_delta_limit", profile.joint_names[joint_index])
        return SafetyDecision(True, "ok")

