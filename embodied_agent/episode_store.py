"""Append-only episode metadata and event logging; media stays in separate files."""

from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
from typing import Any, Mapping

from embodied_agent.contracts import EpisodeManifest


class EpisodeStore:
    """Persists immutable manifests and JSONL events under a run directory."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def write_manifest(self, manifest: EpisodeManifest) -> Path:
        path = self.root / "episodes" / manifest.episode_id / "manifest.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation prevents silently rewriting a recorded episode.
        with path.open("x", encoding="utf-8") as stream:
            json.dump(asdict(manifest), stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        return path

    def append_event(self, event: Mapping[str, Any]) -> None:
        path = self.root / "events.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    dict(event), ensure_ascii=False, sort_keys=True, default=str
                )
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())

