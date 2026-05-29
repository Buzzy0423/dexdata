# Copyright (C) 2026 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Integration tests for the Writer ↔ metadata sidecar wiring.

Covers:
  * Writer with no metadata leaves directory sidecar-free.
  * Writer with metadata writes a sidecar at close().
  * Derivable fields (duration_s, length, size_bytes, checksum, uri) are
    populated by the writer and overwrite caller-supplied values.
  * ``length`` honors the ``observation`` topic slot when present, and
    falls back to the longest stream otherwise.
  * Recorder-context fields (operator_id, task language, robot identity,
    enums) survive the round-trip untouched.
  * Reader.metadata loads the sidecar and is None when absent.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from dexdata.handlers.containers import NumericArraySpec
from dexdata.mcap_utils.reader import Reader
from dexdata.mcap_utils.writer import EPISODE_FILE, Writer
from dexdata.metadata import (
    METADATA_FILE,
    SCHEMA_VERSION,
    Collection,
    Data,
    DataClass,
    EpisodeMetadata,
    Robot,
    Scene,
    Source,
    Task,
    Version,
    load_metadata,
    new_episode_id,
    now_iso,
)
from dexdata.spec import SignalSpec, Spec

BASE_TS_NS = 1_700_000_000_000_000_000


def _two_topic_spec() -> Spec:
    f32 = np.dtype("float32")
    return Spec(
        spec_id="meta_smoke",
        spec_version="0.1.0",
        signals=(
            SignalSpec(
                topic="/robot/state",
                unit="rad",
                container=NumericArraySpec(shape=(7,), dtype=f32),
            ),
            SignalSpec(
                topic="/robot/action",
                unit="rad",
                container=NumericArraySpec(shape=(7,), dtype=f32),
            ),
        ),
    )


def _meta(
    *,
    duration_s: float = 0.0,
    length: int = 0,
    size_bytes: int | None = None,
    checksum_sha256: str | None = None,
    uri: str = "placeholder.mcap",
) -> EpisodeMetadata:
    return EpisodeMetadata(
        version=Version(
            episode_id=new_episode_id(),
            schema_version=SCHEMA_VERSION,
            created_at=now_iso(),
            source=Source.ROBOT_TELEOP,
        ),
        collection=Collection(
            duration_s=duration_s,
            record_hz=20,
            length=length,
            operator_id="test_op",
            data_class=DataClass.EXPERT_DEMO,
            success=True,
        ),
        robot=Robot(
            robot_id="#vega",
            embodiment="vega_1u_gripper",
            data_spec_id="vega_1u_gripper_data_spec_v0",
        ),
        task=Task(language="test task"),
        scene=Scene(),
        data=Data(
            uri=uri,
            topics=[],
            size_bytes=size_bytes,
            checksum_sha256=checksum_sha256,
        ),
    )


def _write_two_topics(
    out_dir: Path,
    spec: Spec,
    *,
    metadata: EpisodeMetadata | None,
    length_topic: str | None = None,
    n_state: int = 50,
    n_action: int = 25,
    state_period_ns: int = 50_000_000,  # 20 Hz
    action_period_ns: int = 100_000_000,  # 10 Hz
) -> None:
    f32 = np.dtype("float32")
    state_arr = np.zeros(7, dtype=f32)
    action_arr = np.ones(7, dtype=f32)
    with Writer(out_dir, spec, metadata=metadata, length_topic=length_topic) as writer:
        for i in range(n_state):
            ts = BASE_TS_NS + i * state_period_ns
            writer.write("/robot/state", state_arr, ts, ts)
        for i in range(n_action):
            ts = BASE_TS_NS + i * action_period_ns
            writer.write("/robot/action", action_arr, ts, ts)


# ---- no-metadata path preserves prior behavior ---------------------------


def test_writer_without_metadata_writes_no_sidecar(tmp_path: Path) -> None:
    out_dir = tmp_path / "ep"
    _write_two_topics(out_dir, _two_topic_spec(), metadata=None)
    assert (out_dir / EPISODE_FILE).exists()
    assert not (out_dir / METADATA_FILE).exists()


# ---- with-metadata: sidecar exists and derivables are populated ----------


def test_writer_writes_sidecar_at_close(tmp_path: Path) -> None:
    out_dir = tmp_path / "ep"
    _write_two_topics(out_dir, _two_topic_spec(), metadata=_meta())
    assert (out_dir / METADATA_FILE).exists()
    loaded = load_metadata(out_dir / METADATA_FILE)
    assert loaded.collection.operator_id == "test_op"
    assert loaded.robot.embodiment == "vega_1u_gripper"


def test_writer_overrides_uri_to_episode_mcap(tmp_path: Path) -> None:
    """Caller-supplied ``uri`` is replaced — writer is authoritative."""
    out_dir = tmp_path / "ep"
    _write_two_topics(
        out_dir, _two_topic_spec(), metadata=_meta(uri="caller_made_this_up.mcap")
    )
    loaded = load_metadata(out_dir / METADATA_FILE)
    assert loaded.data.uri == EPISODE_FILE


def test_writer_populates_size_bytes(tmp_path: Path) -> None:
    out_dir = tmp_path / "ep"
    _write_two_topics(out_dir, _two_topic_spec(), metadata=_meta())
    loaded = load_metadata(out_dir / METADATA_FILE)
    assert loaded.data.size_bytes == (out_dir / EPISODE_FILE).stat().st_size


def test_writer_populates_checksum_sha256(tmp_path: Path) -> None:
    out_dir = tmp_path / "ep"
    _write_two_topics(out_dir, _two_topic_spec(), metadata=_meta())
    loaded = load_metadata(out_dir / METADATA_FILE)
    expected = hashlib.sha256((out_dir / EPISODE_FILE).read_bytes()).hexdigest()
    assert loaded.data.checksum_sha256 == expected
    assert len(loaded.data.checksum_sha256) == 64


def test_writer_populates_duration_s(tmp_path: Path) -> None:
    out_dir = tmp_path / "ep"
    # state: 50 frames @ 20 Hz → last_ts = BASE + 49 * 50ms = +2.45 s
    # action: 25 frames @ 10 Hz → last_ts = BASE + 24 * 100ms = +2.40 s
    # first = BASE, last = BASE + 2.45 s → duration = 2.45 s
    _write_two_topics(out_dir, _two_topic_spec(), metadata=_meta())
    loaded = load_metadata(out_dir / METADATA_FILE)
    assert abs(loaded.collection.duration_s - 2.45) < 1e-9


# ---- length: length_topic param vs longest-stream fallback ----------------


def test_length_uses_length_topic_param(tmp_path: Path) -> None:
    out_dir = tmp_path / "ep"
    # 50 state, 25 action. length_topic=action → length=25.
    _write_two_topics(
        out_dir,
        _two_topic_spec(),
        metadata=_meta(),
        length_topic="/robot/action",
    )
    loaded = load_metadata(out_dir / METADATA_FILE)
    assert loaded.collection.length == 25


def test_length_falls_back_to_longest_stream(tmp_path: Path) -> None:
    out_dir = tmp_path / "ep"
    # No length_topic → fallback picks the longest stream (state=50).
    _write_two_topics(out_dir, _two_topic_spec(), metadata=_meta())
    loaded = load_metadata(out_dir / METADATA_FILE)
    assert loaded.collection.length == 50


def test_length_falls_back_when_length_topic_unwritten(tmp_path: Path) -> None:
    """length_topic names a topic that was never written → fallback."""
    out_dir = tmp_path / "ep"
    _write_two_topics(
        out_dir,
        _two_topic_spec(),
        metadata=_meta(),
        length_topic="/robot/never_published",
    )
    loaded = load_metadata(out_dir / METADATA_FILE)
    assert loaded.collection.length == 50  # longest = state


def test_topics_list_populated_by_writer(tmp_path: Path) -> None:
    """``data.topics`` reflects every topic that received messages."""
    out_dir = tmp_path / "ep"
    _write_two_topics(out_dir, _two_topic_spec(), metadata=_meta())
    loaded = load_metadata(out_dir / METADATA_FILE)
    assert loaded.data.topics == ["/robot/action", "/robot/state"]


# ---- recorder-context fields survive untouched ----------------------------


def test_recorder_context_fields_preserved(tmp_path: Path) -> None:
    out_dir = tmp_path / "ep"
    meta = _meta()
    original_episode_id = meta.version.episode_id
    original_created_at = meta.version.created_at
    original_operator = meta.collection.operator_id
    original_embodiment = meta.robot.embodiment

    _write_two_topics(out_dir, _two_topic_spec(), metadata=meta)
    loaded = load_metadata(out_dir / METADATA_FILE)
    assert loaded.version.episode_id == original_episode_id
    assert loaded.version.created_at == original_created_at
    assert loaded.collection.operator_id == original_operator
    assert loaded.robot.embodiment == original_embodiment
    assert loaded.version.source == Source.ROBOT_TELEOP
    assert loaded.collection.data_class == DataClass.EXPERT_DEMO


# ---- Reader.metadata accessor ---------------------------------------------


def test_reader_metadata_loads(tmp_path: Path) -> None:
    out_dir = tmp_path / "ep"
    _write_two_topics(out_dir, _two_topic_spec(), metadata=_meta())
    reader = Reader(out_dir)
    assert reader.metadata is not None
    assert reader.metadata.robot.embodiment == "vega_1u_gripper"


def test_reader_metadata_is_none_when_sidecar_absent(tmp_path: Path) -> None:
    out_dir = tmp_path / "ep"
    _write_two_topics(out_dir, _two_topic_spec(), metadata=None)
    reader = Reader(out_dir)
    assert reader.metadata is None


def test_reader_metadata_is_cached(tmp_path: Path) -> None:
    """Second access doesn't re-read from disk."""
    out_dir = tmp_path / "ep"
    _write_two_topics(out_dir, _two_topic_spec(), metadata=_meta())
    reader = Reader(out_dir)
    first = reader.metadata
    # Delete the sidecar — cached value should still come back.
    (out_dir / METADATA_FILE).unlink()
    second = reader.metadata
    assert first is second


# ---- empty episode (no writes) --------------------------------------------


def test_empty_episode_finalizes_zero_duration_and_length(tmp_path: Path) -> None:
    out_dir = tmp_path / "ep"
    with Writer(out_dir, _two_topic_spec(), metadata=_meta()):
        pass  # no writes
    loaded = load_metadata(out_dir / METADATA_FILE)
    assert loaded.collection.duration_s == 0.0
    assert loaded.collection.length == 0
    assert loaded.data.size_bytes is not None
    assert loaded.data.size_bytes > 0  # MCAP header still present
