"""Deterministic routing to a policy that explicitly supports the robot."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from embodied_agent.interfaces import VLAAdapter


@dataclass(frozen=True, slots=True)
class ModelCard:
    model_id: str
    embodiments: frozenset[str]
    modalities: frozenset[str]
    action_schemas: frozenset[str]
    skills: frozenset[str] = frozenset()
    priority: int = 0


class PolicyRouter:
    """Selects a ready, schema-compatible model; never guesses compatibility."""

    def __init__(self) -> None:
        self._models: dict[str, tuple[ModelCard, VLAAdapter]] = {}

    def register(self, card: ModelCard, adapter: VLAAdapter) -> None:
        if card.model_id != adapter.model_id:
            raise ValueError("Model card ID must match adapter.model_id")
        if card.model_id in self._models:
            raise ValueError(f"Model already registered: {card.model_id}")
        self._models[card.model_id] = (card, adapter)

    def select(
        self,
        *,
        embodiment_id: str,
        modalities: Iterable[str],
        action_schema: str,
        skill_id: str,
        preferred: Iterable[str] = (),
    ) -> VLAAdapter:
        required_modalities = frozenset(modalities)
        preferred_rank = {model_id: rank for rank, model_id in enumerate(preferred)}
        compatible = []
        for card, adapter in self._models.values():
            if embodiment_id not in card.embodiments:
                continue
            if action_schema not in card.action_schemas:
                continue
            if not required_modalities.issubset(card.modalities):
                continue
            if card.skills and skill_id not in card.skills:
                continue
            if adapter.is_ready():
                compatible.append((card, adapter))
        if not compatible:
            raise LookupError(
                f"No ready VLA supports embodiment={embodiment_id!r}, "
                f"schema={action_schema!r}, modalities={sorted(required_modalities)!r}"
            )
        compatible.sort(
            key=lambda pair: (
                preferred_rank.get(pair[0].model_id, len(preferred_rank)),
                -pair[0].priority,
                pair[0].model_id,
            )
        )
        return compatible[0][1]

