# Copyright (C) 2026 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Tests for the composable :mod:`metadata` sidecar."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from dexdata.metadata import (
    SCHEMA_VERSION,
    Collection,
    Data,
    DataClass,
    DextraceEntry,
    EpisodeMetadata,
    FailureMode,
    Hand,
    Robot,
    Scene,
    Source,
    Task,
    TeleopMethod,
    Version,
    load_metadata,
    metadata_from_dict,
    new_episode_id,
    now_iso,
    save_metadata,
)


def _minimal_meta(
    *,
    source: Source = Source.ROBOT_TELEOP,
    success: bool = True,
    failure_mode: FailureMode | None = None,
    dextrace: list[DextraceEntry] | None = None,
    data_class: DataClass = DataClass.EXPERT_DEMO,
    robot: Robot | None = None,
) -> EpisodeMetadata:
    return EpisodeMetadata(
        version=Version(
            episode_id=new_episode_id(),
            schema_version=SCHEMA_VERSION,
            created_at=now_iso(),
            source=source,
        ),
        collection=Collection(
            duration_s=6.2,
            record_hz=20,
            length=124,
            operator_id="test_user",
            data_class=data_class,
            success=success,
            failure_mode=failure_mode,
        ),
        robot=robot
        if robot is not None
        else Robot(
            robot_id="#vega",
            embodiment="vega_1u_gripper",
            data_spec_id="vega_1u_gripper_data_spec_v0",
        ),
        task=Task(language="pick up the cube"),
        scene=Scene(),
        data=Data(
            uri="episode.mcap",
            topics=[
                "/robot/action/left_arm/qpos",
                "/robot/state/left_arm/qpos",
                "/camera/head_left/rgb/video",
            ],
            checksum_sha256="ab" * 32,
            size_bytes=12345,
        ),
        dextrace=dextrace,
    )


# ---- round-trip ----------------------------------------------------------


def test_save_load_round_trip(tmp_path: Path) -> None:
    meta = _minimal_meta()
    path = save_metadata(meta, tmp_path / "metadata.json")
    assert path.exists()
    loaded = load_metadata(path)
    assert loaded.to_dict() == meta.to_dict()


def test_enums_serialize_as_strings(tmp_path: Path) -> None:
    meta = _minimal_meta()
    save_metadata(meta, tmp_path / "metadata.json")
    raw = json.loads((tmp_path / "metadata.json").read_text())
    assert raw["version"]["source"] == "robot_teleop"
    assert raw["collection"]["data_class"] == "expert_demo"


def test_optional_enums_round_trip(tmp_path: Path) -> None:
    meta = _minimal_meta(success=False, failure_mode=FailureMode.SLIP)
    meta.collection.teleop_method = TeleopMethod.VR
    path = save_metadata(meta, tmp_path / "metadata.json")
    loaded = load_metadata(path)
    assert loaded.collection.failure_mode == FailureMode.SLIP
    assert loaded.collection.teleop_method == TeleopMethod.VR


def test_topics_list_round_trips(tmp_path: Path) -> None:
    meta = _minimal_meta()
    path = save_metadata(meta, tmp_path / "metadata.json")
    loaded = load_metadata(path)
    assert loaded.data.topics == [
        "/robot/action/left_arm/qpos",
        "/robot/state/left_arm/qpos",
        "/camera/head_left/rgb/video",
    ]


def test_dextrace_round_trip(tmp_path: Path) -> None:
    meta = _minimal_meta(
        source=Source.DEXTRACE,
        dextrace=[DextraceEntry(hand=Hand.LEFT), DextraceEntry(hand=Hand.RIGHT)],
    )
    path = save_metadata(meta, tmp_path / "metadata.json")
    raw = json.loads(path.read_text())
    assert raw["dextrace"] == [{"hand": "left"}, {"hand": "right"}]
    loaded = load_metadata(path)
    assert [e.hand for e in loaded.dextrace] == [Hand.LEFT, Hand.RIGHT]


def test_dextrace_missing_serializes_as_null(tmp_path: Path) -> None:
    """Non-dextrace episodes render the field as JSON null, not ``[]``."""
    meta = _minimal_meta()  # default source=ROBOT_TELEOP, dextrace=None
    assert meta.dextrace is None
    save_metadata(meta, tmp_path / "metadata.json")
    raw = json.loads((tmp_path / "metadata.json").read_text())
    assert raw["dextrace"] is None
    # Round-trip preserves None (not silently coerced to []).
    loaded = load_metadata(tmp_path / "metadata.json")
    assert loaded.dextrace is None


def test_dextrace_loads_null_as_none() -> None:
    raw = _minimal_meta().to_dict()
    raw["dextrace"] = None
    meta = metadata_from_dict(raw)
    assert meta.dextrace is None


# ---- invariants ----------------------------------------------------------


def test_dextrace_coupling_dextrace_source_without_entries_rejected() -> None:
    with pytest.raises(ValueError, match="coupling violation"):
        _minimal_meta(source=Source.DEXTRACE)


def test_dextrace_coupling_entries_without_dextrace_source_rejected() -> None:
    with pytest.raises(ValueError, match="coupling violation"):
        _minimal_meta(dextrace=[DextraceEntry(hand=Hand.LEFT)])


def test_dextrace_coupling_satisfied_constructs() -> None:
    meta = _minimal_meta(
        source=Source.DEXTRACE,
        dextrace=[DextraceEntry(hand=Hand.LEFT)],
    )
    assert meta.version.source == Source.DEXTRACE
    assert len(meta.dextrace) == 1


def test_success_with_failure_mode_rejected() -> None:
    with pytest.raises(ValueError, match="failure_mode"):
        _minimal_meta(success=True, failure_mode=FailureMode.TIMEOUT)


def test_failure_without_failure_mode_allowed() -> None:
    # Success=False with failure_mode=None is permitted — the failure
    # mode just hasn't been triaged yet.
    meta = _minimal_meta(success=False, failure_mode=None)
    assert meta.collection.success is False
    assert meta.collection.failure_mode is None


def test_schema_version_mismatch_rejected(tmp_path: Path) -> None:
    raw = _minimal_meta().to_dict()
    raw["version"]["schema_version"] = "0.0.0"
    with pytest.raises(ValueError, match="schema_version"):
        metadata_from_dict(raw)


def test_unknown_enum_value_rejected() -> None:
    raw = _minimal_meta().to_dict()
    raw["version"]["source"] = "made_up_source"
    with pytest.raises(ValueError):  # Enum() raises ValueError on miss
        metadata_from_dict(raw)


# ---- shape coverage ------------------------------------------------------


def test_human_video_allows_all_robot_fields_none() -> None:
    meta = _minimal_meta(
        data_class=DataClass.HUMAN_VIDEO,
        robot=Robot(),
    )
    assert meta.robot.embodiment is None
    assert meta.robot.data_spec_id is None


def test_hardware_revision_is_free_dict(tmp_path: Path) -> None:
    meta = _minimal_meta()
    meta.robot.hardware_revision = {
        "arm_left": "rev_b",
        "arm_right": "rev_b",
        "gripper": "rev_a",
    }
    path = save_metadata(meta, tmp_path / "metadata.json")
    loaded = load_metadata(path)
    assert loaded.robot.hardware_revision == {
        "arm_left": "rev_b",
        "arm_right": "rev_b",
        "gripper": "rev_a",
    }


def test_uuidv7_is_chronologically_sortable() -> None:
    a = new_episode_id()
    time.sleep(0.01)
    b = new_episode_id()
    # UUIDv7's leading 48 bits are a Unix millisecond timestamp, so
    # lexicographic order on the hex string == chronological order.
    assert a < b


def test_atomic_write_no_tmp_leftover(tmp_path: Path) -> None:
    path = save_metadata(_minimal_meta(), tmp_path / "metadata.json")
    siblings = list(path.parent.iterdir())
    assert siblings == [path], f"unexpected sibling files: {siblings}"
