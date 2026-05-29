# Copyright (C) 2026 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Episode metadata sidecar for the composable layer.

Each composable episode directory may carry a ``metadata.json`` sidecar
alongside its ``episode.mcap``. The sidecar holds recorder-time context
that isn't derivable from the MCAP itself (operator, task, robot
identity, source kind, ...) plus a small set of derivable fields
persisted for catalog-scale queryability (``duration_s``, ``length``,
``size_bytes``, ``checksum_sha256``) — so a fleet of episodes can be
scanned without opening any MCAP.

The schema is intentionally a clean cut from the legacy
``dexdata.mcap_utils.metadata`` format: no readback compatibility, no
field-by-field migration. Converters can be written separately if a
downstream tool still depends on the 0.3.0 shape.

This module provides:

* :class:`EpisodeMetadata` and its sub-dataclasses — the canonical
  Python view.
* Enums for closed-vocabulary fields (:class:`Source`,
  :class:`TeleopMethod`, :class:`DataClass`, :class:`FailureMode`,
  :class:`Hand`). All are ``str`` subclasses so JSON serializes them
  as their string value with no custom encoder.
* :func:`save_metadata` / :func:`load_metadata` — JSON file helpers.
* :func:`new_episode_id` / :func:`now_iso` — small helpers for the
  ``Version`` block.

Invariants enforced at construction time:

* ``Source.DEXTRACE`` ⇔ non-empty ``dextrace`` list (biconditional
  coupling). For non-dextrace sources the field is ``None`` and
  serializes as JSON ``null`` — distinct from an empty list, which
  would imply "we have a dextrace section, it just has no entries".
* ``Collection.success`` ⇒ ``failure_mode is None``.
* ``Version.schema_version`` must equal :data:`SCHEMA_VERSION`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import UTC, datetime
from enum import Enum, StrEnum
from pathlib import Path
from typing import Any

import uuid6

SCHEMA_VERSION = "0.1.0"
METADATA_FILE = "metadata.json"


# ---- enums ---------------------------------------------------------------
#
# StrEnum members are strings at runtime — ``json.dumps`` serializes
# ``Source.ROBOT_TELEOP`` directly as ``"robot_teleop"`` with no custom
# encoder, and ``Source("robot_teleop")`` rehydrates it on load.


class Source(StrEnum):
    ROBOT_TELEOP = "robot_teleop"
    ROBOT_SIM = "robot_sim"
    DEXTRACE = "dextrace"


class TeleopMethod(StrEnum):
    EXOSKELETON = "exoskeleton"
    VR = "vr"
    UMI = "umi"


class DataClass(StrEnum):
    EXPERT_DEMO = "expert_demo"
    DEMO_LOW_QUALITY = "demo_low_quality"
    AUTONOMOUS_SUCCESS = "autonomous_success"
    AUTONOMOUS_FAILURE = "autonomous_failure"
    INTERVENTION = "intervention"
    RL_ROLLOUT = "rl_rollout"
    SCRIPTED = "scripted"
    HUMAN_VIDEO = "human_video"


class FailureMode(StrEnum):
    SLIP = "slip"
    COLLISION = "collision"
    WRONG_OBJECT = "wrong_object"
    TIMEOUT = "timeout"
    DROPPED = "dropped"
    MISSED_GRASP = "missed_grasp"
    POSE_UNREACHABLE = "pose_unreachable"
    OTHER = "other"


class Hand(StrEnum):
    LEFT = "left"
    RIGHT = "right"


# ---- dataclasses ---------------------------------------------------------


@dataclass
class Version:
    episode_id: str  # UUIDv7 hex (see new_episode_id())
    schema_version: str  # must equal SCHEMA_VERSION
    created_at: str  # ISO-8601 UTC with "Z" suffix
    source: Source


@dataclass
class Collection:
    duration_s: float
    record_hz: int
    length: int  # observation-stream frame count
    operator_id: str
    data_class: DataClass
    success: bool
    collection_protocol: str | None = None
    teleop_method: TeleopMethod | None = None
    control_hz: int | None = None
    failure_mode: FailureMode | None = None


@dataclass
class Robot:
    # All-optional: human_video episodes legitimately leave every field None.
    robot_id: str | None = None
    embodiment: str | None = None  # joins to composable Spec.spec_id
    hardware_revision: dict[str, Any] | None = None
    firmware_version: str | None = None
    urdf_id: str | None = None
    calibration_id: str | None = None
    data_spec_id: str | None = None
    sensors_spec_id: str | None = None


@dataclass
class DextraceEntry:
    hand: Hand


@dataclass
class Task:
    task_id: str | None = None
    language: str = ""
    language_alt: list[str] = field(default_factory=list)


@dataclass
class Scene:
    description: str | None = None


@dataclass
class Data:
    uri: str  # path to the MCAP, relative to the sidecar
    # Flat list of topic strings actually written to ``uri``. Populated
    # by Writer at close from the topics that received messages. No
    # null/dummy entries — the slot-vocabulary approach was dropped
    # because it hid bilateral state behind single "observation" /
    # "action" representatives, while leaving placeholder nulls for
    # capabilities not on the embodiment.
    topics: list[str] = field(default_factory=list)
    checksum_sha256: str | None = None
    size_bytes: int | None = None


@dataclass
class EpisodeMetadata:
    version: Version
    collection: Collection
    robot: Robot
    task: Task
    scene: Scene
    data: Data
    # ``None`` for non-dextrace sources — serializes as JSON ``null`` so
    # the field is visibly N/A rather than "empty section". An empty
    # list would also satisfy the biconditional below, but conflates
    # "no dextrace info applies" with "dextrace section present, no
    # entries observed".
    dextrace: list[DextraceEntry] | None = None

    def __post_init__(self) -> None:
        if self.version.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"schema_version {self.version.schema_version!r} != "
                f"supported {SCHEMA_VERSION!r}"
            )
        # Source.DEXTRACE ⇔ dextrace has ≥1 entries. Strict biconditional
        # — a dextrace-sourced episode without entries (or vice versa) is
        # treated as a recorder bug, not a tolerated configuration. Both
        # ``None`` and ``[]`` count as "no entries" for this check.
        is_dextrace_source = self.version.source == Source.DEXTRACE
        has_dextrace = bool(self.dextrace)
        if is_dextrace_source != has_dextrace:
            n_entries = 0 if self.dextrace is None else len(self.dextrace)
            raise ValueError(
                f"coupling violation: version.source="
                f"{self.version.source.value!r} but dextrace has "
                f"{n_entries} entries; expected "
                f"{'>0' if is_dextrace_source else '0'}"
            )
        if self.collection.success and self.collection.failure_mode is not None:
            raise ValueError(
                f"failure_mode={self.collection.failure_mode.value!r} is "
                f"set but success=True; failure_mode is only meaningful "
                f"when success=False"
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-ready nested dict."""
        return _dataclass_to_dict(self)


# ---- JSON I/O ------------------------------------------------------------


def metadata_from_dict(d: dict[str, Any]) -> EpisodeMetadata:
    """Rehydrate :class:`EpisodeMetadata` from a JSON-deserialized dict.

    Enum fields are constructed via their value (e.g.
    ``Source("robot_teleop")``); unknown values raise ``ValueError``.
    Missing optional fields default per the dataclass definitions.
    """
    v = d["version"]
    c = d["collection"]
    r = d["robot"]
    t = d["task"]
    s = d["scene"]
    da = d["data"]
    return EpisodeMetadata(
        version=Version(
            episode_id=v["episode_id"],
            schema_version=v["schema_version"],
            created_at=v["created_at"],
            source=Source(v["source"]),
        ),
        collection=Collection(
            duration_s=float(c["duration_s"]),
            record_hz=int(c["record_hz"]),
            length=int(c["length"]),
            operator_id=c["operator_id"],
            data_class=DataClass(c["data_class"]),
            success=bool(c["success"]),
            collection_protocol=c.get("collection_protocol"),
            teleop_method=_opt_enum(c.get("teleop_method"), TeleopMethod),
            control_hz=c.get("control_hz"),
            failure_mode=_opt_enum(c.get("failure_mode"), FailureMode),
        ),
        robot=Robot(
            robot_id=r.get("robot_id"),
            embodiment=r.get("embodiment"),
            hardware_revision=r.get("hardware_revision"),
            firmware_version=r.get("firmware_version"),
            urdf_id=r.get("urdf_id"),
            calibration_id=r.get("calibration_id"),
            data_spec_id=r.get("data_spec_id"),
            sensors_spec_id=r.get("sensors_spec_id"),
        ),
        task=Task(
            task_id=t.get("task_id"),
            language=t.get("language", ""),
            language_alt=list(t.get("language_alt", [])),
        ),
        scene=Scene(description=s.get("description")),
        data=Data(
            uri=da["uri"],
            topics=list(da.get("topics", [])),
            checksum_sha256=da.get("checksum_sha256"),
            size_bytes=da.get("size_bytes"),
        ),
        # Missing key or explicit null both load as None; a list (even
        # empty) is preserved as-is. The invariant check in
        # __post_init__ then rejects [] from a DEXTRACE source.
        dextrace=(
            None
            if d.get("dextrace") is None
            else [DextraceEntry(hand=Hand(e["hand"])) for e in d["dextrace"]]
        ),
    )


def save_metadata(meta: EpisodeMetadata, path: Path | str) -> Path:
    """Atomically write ``meta`` as JSON to ``path``.

    Writes to a sibling ``.tmp`` file first and renames, so a partial
    sidecar can never appear on disk under the final name.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(meta.to_dict(), indent=2) + "\n")
    tmp.replace(path)
    return path


def load_metadata(path: Path | str) -> EpisodeMetadata:
    """Load and validate a sidecar at ``path``."""
    with open(path) as f:
        return metadata_from_dict(json.load(f))


# ---- helpers -------------------------------------------------------------


def new_episode_id() -> str:
    """Fresh UUIDv7 hex string — sortable by embedded millisecond timestamp."""
    return uuid6.uuid7().hex


def now_iso() -> str:
    """ISO-8601 UTC timestamp with ``Z`` suffix and microsecond precision."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def compute_file_checksum_and_size(path: Path | str) -> tuple[str, int]:
    """Streaming SHA-256 hex digest + byte size of ``path``.

    Used by the writer to finalize ``Data.checksum_sha256`` /
    ``Data.size_bytes`` after the MCAP is closed.
    """
    import hashlib

    h = hashlib.sha256()
    size = 0
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
            size += len(chunk)
    return h.hexdigest(), size


def _opt_enum(value: str | None, cls: type[Enum]) -> Any:
    return None if value is None else cls(value)


def _dataclass_to_dict(obj: Any) -> Any:
    """Recursively render a dataclass tree to JSON-ready primitives.

    Enum members are serialized as their ``.value``. ``None`` is
    preserved (the sidecar uses ``null`` as a meaningful sentinel for
    optional fields and unpopulated topic slots).
    """
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _dataclass_to_dict(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, list):
        return [_dataclass_to_dict(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _dataclass_to_dict(v) for k, v in obj.items()}
    return obj
