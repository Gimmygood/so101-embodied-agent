"""Single-step, fail-closed orchestration for a physical robot episode."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

from embodied_agent.contracts import Observation
from embodied_agent.episode_store import EpisodeStore
from embodied_agent.interfaces import (
    Brain,
    DemoMemory,
    HumanApproval,
    RobotBackend,
)
from embodied_agent.policy_router import PolicyRouter
from embodied_agent.safety import SafetySupervisor
from embodied_agent.skill_registry import SkillRegistry


@dataclass(frozen=True, slots=True)
class StepReport:
    status: str
    skill_id: str | None = None
    policy_id: str | None = None
    safety_code: str | None = None
    detail: str = ""


class AgentRuntime:
    """One fresh observation → one approved skill → one bounded action chunk.

    The caller owns the outer task loop and operator UI. This method never
    repeats a physical command after an uncertain execution result.
    """

    def __init__(
        self,
        *,
        brain: Brain,
        memory: DemoMemory,
        skills: SkillRegistry,
        policies: PolicyRouter,
        robot: RobotBackend,
        safety: SafetySupervisor,
        human: HumanApproval,
        store: EpisodeStore,
        preferred_policies: tuple[str, ...] = (),
        demo_limit: int = 3,
    ) -> None:
        self.brain = brain
        self.memory = memory
        self.skills = skills
        self.policies = policies
        self.robot = robot
        self.safety = safety
        self.human = human
        self.store = store
        self.preferred_policies = preferred_policies
        self.demo_limit = demo_limit

    def step(self, goal: str) -> StepReport:
        try:
            healthy, health_reason = self.robot.health()
        except Exception as error:
            self.robot.safe_stop("robot_health_check_failed")
            return StepReport(
                "needs_human", safety_code="robot_health_check_failed", detail=str(error)
            )
        if not healthy:
            self.robot.safe_stop("robot_not_ready")
            return StepReport("needs_human", safety_code="robot_not_ready", detail=health_reason)
        try:
            observation = self.robot.observe()
        except Exception as error:
            self.robot.safe_stop("observation_failed")
            return StepReport("needs_human", safety_code="observation_failed", detail=str(error))
        manifests = self.skills.available(observation.embodiment_id)
        if not manifests:
            return self._pause(observation, "no_verified_skill")
        try:
            demonstrations = self.memory.retrieve(
                goal, observation, limit=self.demo_limit
            )
            call = self.brain.choose_skill(
                goal, observation, demonstrations, manifests
            )
        except Exception as error:
            return self._pause(observation, "planning_failed", str(error))
        try:
            skill = self.skills.resolve(call.skill_id, observation.embodiment_id)
        except LookupError as error:
            return self._pause(observation, "unregistered_skill", str(error))

        manifest = skill.manifest
        if call.request_human_approval or manifest.requires_human_approval:
            if not self.human.request(
                f"Approval required for skill {manifest.skill_id}@{manifest.version}",
                observation,
            ):
                return self._pause(observation, "human_declined", manifest.skill_id)

        modalities = observation.metadata.get("modalities", ())
        try:
            policy = self.policies.select(
                embodiment_id=observation.embodiment_id,
                modalities=modalities,
                action_schema=manifest.action_schema,
                skill_id=manifest.skill_id,
                preferred=self.preferred_policies,
            )
        except LookupError as error:
            return self._pause(observation, "no_compatible_policy", str(error))

        proposal_started = time.monotonic()
        try:
            result = skill.propose(call, observation, policy)
        except Exception as error:
            self.robot.safe_stop("skill_proposal_failed")
            self._event(observation, "skill_proposal_failed", error=repr(error))
            return StepReport("needs_human", manifest.skill_id, policy.model_id, detail=str(error))
        if time.monotonic() - proposal_started > manifest.max_duration_s:
            self.robot.safe_stop("skill_deadline_exceeded")
            self._event(
                observation,
                "skill_deadline_exceeded",
                skill_id=manifest.skill_id,
                policy_id=policy.model_id,
            )
            return StepReport(
                "needs_human",
                manifest.skill_id,
                policy.model_id,
                "skill_deadline_exceeded",
            )

        if result.status != "proposed" or result.action is None:
            self._event(
                observation,
                "skill_did_not_propose_action",
                status=result.status,
                failure_code=result.failure_code,
            )
            return StepReport(
                "needs_human" if result.status == "needs_human" else result.status,
                manifest.skill_id,
                policy.model_id,
                detail=result.failure_code or result.status,
            )

        if result.action.policy_id != policy.model_id:
            return self._pause(observation, "policy_identity_mismatch")

        decision = self.safety.validate(
            observation, result.action, now_ns=time.monotonic_ns()
        )
        if not decision.allowed:
            self.robot.safe_stop(decision.code)
            self._event(
                observation,
                "action_rejected",
                skill_id=manifest.skill_id,
                policy_id=policy.model_id,
                safety_code=decision.code,
                detail=decision.detail,
            )
            return StepReport(
                "needs_human", manifest.skill_id, policy.model_id, decision.code
            )

        try:
            receipt = self.robot.execute(result.action)
        except Exception as error:
            # Execution may have partially happened. Never replay blindly.
            self.robot.safe_stop("execution_receipt_unknown")
            self._event(
                observation,
                "execution_receipt_unknown",
                skill_id=manifest.skill_id,
                policy_id=policy.model_id,
                error=repr(error),
            )
            return StepReport(
                "needs_human",
                manifest.skill_id,
                policy.model_id,
                "execution_receipt_unknown",
                str(error),
            )

        self._event(
            observation,
            "action_executed",
            skill_id=manifest.skill_id,
            skill_version=manifest.version,
            policy_id=policy.model_id,
            receipt=dict(receipt),
        )
        return StepReport("executed", manifest.skill_id, policy.model_id)

    def _pause(
        self, observation: Observation, code: str, detail: str = ""
    ) -> StepReport:
        self.robot.safe_stop(code)
        self._event(observation, code, detail=detail)
        return StepReport("needs_human", safety_code=code, detail=detail)

    def _event(self, observation: Observation, event_type: str, **details: Any) -> None:
        self.store.append_event(
            {
                "run_id": observation.run_id,
                "episode_id": observation.episode_id,
                "observed_at_ns": observation.observed_at_ns,
                "event_type": event_type,
                **details,
            }
        )

