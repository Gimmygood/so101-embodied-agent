"""LeRobot-compatible SO-101 backend boundary.

The transport is injected because LeRobot versions, cameras and serial ports
are machine-specific. This adapter normalizes the transport to the agent
contract; it does not discover devices or infer calibration automatically.
"""

from __future__ import annotations

from typing import Mapping, Protocol

from embodied_agent.contracts import ActionChunk, Observation


class SO101Transport(Protocol):
    def health(self) -> tuple[bool, str]: ...

    def read_observation(self) -> Observation: ...

    def write_joint_targets(
        self,
        joint_names: tuple[str, ...],
        positions: tuple[float, ...],
        *,
        timeout_ms: int,
    ) -> Mapping[str, object]: ...

    def safe_stop(self, reason: str) -> None: ...


class SO101Backend:
    """Adapter for one calibrated SO-101 transport and one action schema."""

    def __init__(
        self,
        transport: SO101Transport,
        *,
        calibration_id: str,
        action_schema: str,
        command_timeout_ms: int = 200,
    ) -> None:
        if not calibration_id or not action_schema:
            raise ValueError("SO-101 calibration and action schema are required")
        self.transport = transport
        self.calibration_id = calibration_id
        self.action_schema = action_schema
        self.command_timeout_ms = command_timeout_ms

    def health(self) -> tuple[bool, str]:
        return self.transport.health()

    def observe(self) -> Observation:
        observation = self.transport.read_observation()
        if observation.embodiment_id != "so101":
            raise ValueError("Connected transport did not identify as SO-101")
        if observation.calibration_id != self.calibration_id:
            raise ValueError("SO-101 calibration changed; stop and re-approve")
        return observation

    def execute(self, action: ActionChunk) -> Mapping[str, object]:
        if action.action_schema != self.action_schema:
            raise ValueError("Action schema does not match the SO-101 profile")
        if len(action.steps) != 1:
            raise ValueError("Initial SO-101 backend accepts one action step only")
        return self.transport.write_joint_targets(
            action.joint_names,
            action.steps[0],
            timeout_ms=self.command_timeout_ms,
        )

    def safe_stop(self, reason: str) -> None:
        self.transport.safe_stop(reason)

