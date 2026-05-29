# Copyright (C) 2026 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Recover an unclosed / truncated composable MCAP in place.

Streams records front-to-back from the truncated source (no footer seek)
into a fresh ``McapWriter``, stops at the first parse error or end of
stream, calls ``finish()`` so the result has a real summary, then atomic-
replaces the original ``episode.mcap`` and regenerates the derivable
fields of ``metadata.json`` (the recorder-time fields — operator, task,
embodiment — are preserved).

Caveat: this is a generic record-level copy. ``CompressedVideo`` channels
whose stream was cut mid-GOP will produce a closed file whose trailing
frames can't decode until the next keyframe; consumers will see fewer
decoded frames than messages. Trimming partial GOPs requires a video-
decode pass and is intentionally out of scope here.

Idempotent: running on an already-closed episode rewrites it byte-equiv
and refreshes the sidecar.

Usage::

    python scripts/recover_episode.py --path <episode_dir>
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import tyro
from mcap.exceptions import McapError
from mcap.records import Channel, Header, Message, Schema
from mcap.stream_reader import StreamReader
from mcap.writer import Writer as McapWriter

from dexdata import EPISODE_FILE
from dexdata.metadata import (
    METADATA_FILE,
    compute_file_checksum_and_size,
    load_metadata,
    save_metadata,
)

RECOVERY_TMP_SUFFIX = ".recover.tmp"


def main(path: Path) -> None:
    """Recover the episode at ``path`` in place.

    Args:
        path: An episode directory containing ``episode.mcap``.
    """
    if not path.is_dir():
        print(f"not a directory: {path}", file=sys.stderr)
        sys.exit(2)
    src_mcap = path / EPISODE_FILE
    if not src_mcap.exists():
        print(f"missing {EPISODE_FILE} in {path}", file=sys.stderr)
        sys.exit(2)

    tmp_mcap = path / (EPISODE_FILE + RECOVERY_TMP_SUFFIX)
    if tmp_mcap.exists():
        # Leftover from a previous interrupted recovery. Safe to remove —
        # the source mcap is what we read from.
        tmp_mcap.unlink()

    n_msgs, topic_counts, first_log_ns, last_log_ns, spec_ids = _copy_records(
        src_mcap, tmp_mcap
    )

    if n_msgs == 0:
        # Nothing recovered. Leave the source untouched so the user can
        # decide what to do; an empty mcap is worse than a broken one.
        tmp_mcap.unlink(missing_ok=True)
        print(
            f"recovered 0 messages from {src_mcap}; leaving source unchanged",
            file=sys.stderr,
        )
        sys.exit(1)

    # Atomically replace the source mcap with the recovered one. From
    # this point the on-disk mcap is closed and self-consistent; if we
    # crash before the sidecar update, re-running recovery is a no-op
    # that just refreshes the sidecar.
    os.replace(tmp_mcap, src_mcap)

    _refresh_sidecar(path, src_mcap, topic_counts, first_log_ns, last_log_ns, spec_ids)

    print(
        f"recovered {n_msgs} messages across {len(topic_counts)} topics into {src_mcap}"
    )


def _copy_records(
    src_mcap: Path, dst_mcap: Path
) -> tuple[int, dict[str, int], int | None, int | None, set[tuple[str, str]]]:
    """Stream records from ``src_mcap`` into a fresh closed mcap at ``dst_mcap``.

    Returns counters needed to refresh the sidecar:
    ``(n_msgs, {topic: count}, first_log_ns, last_log_ns, {(spec_id, spec_version)})``.
    """
    topic_counts: dict[str, int] = {}
    first_log_ns: int | None = None
    last_log_ns: int | None = None
    # Channel records carry spec identity in their metadata; collect it
    # so the sidecar's ``robot.data_spec_id`` can be refreshed in case
    # the source sidecar pre-dates the writer's spec-derivation logic.
    spec_ids: set[tuple[str, str]] = set()
    # Map source-side ids → freshly-allocated writer-side ids. The
    # writer assigns its own ids on register_*; preserving source ids
    # isn't possible through the public API.
    src_to_dst_schema: dict[int, int] = {}
    src_to_dst_channel: dict[int, int] = {}
    src_id_to_topic: dict[int, str] = {}

    dst_file = open(dst_mcap, "wb")
    writer = McapWriter(dst_file)
    writer.start()

    n_msgs = 0
    try:
        with open(src_mcap, "rb") as f:
            stream = StreamReader(f)
            try:
                for rec in stream.records:
                    if isinstance(rec, Header):
                        # The writer emits its own Header from start().
                        continue
                    if isinstance(rec, Schema):
                        # Closed mcaps repeat Schema/Channel records in
                        # the summary section. Dedupe by src id so we
                        # don't double-register them in the output.
                        if rec.id in src_to_dst_schema:
                            continue
                        new_id = writer.register_schema(
                            name=rec.name,
                            encoding=rec.encoding,
                            data=rec.data,
                        )
                        src_to_dst_schema[rec.id] = new_id
                    elif isinstance(rec, Channel):
                        if rec.id in src_to_dst_channel:
                            continue
                        new_id = writer.register_channel(
                            topic=rec.topic,
                            schema_id=src_to_dst_schema[rec.schema_id],
                            message_encoding=rec.message_encoding,
                            metadata=dict(rec.metadata),
                        )
                        src_to_dst_channel[rec.id] = new_id
                        src_id_to_topic[rec.id] = rec.topic
                        sid = rec.metadata.get("spec_id")
                        sver = rec.metadata.get("spec_version")
                        if sid and sver:
                            spec_ids.add((sid, sver))
                    elif isinstance(rec, Message):
                        dst_chan = src_to_dst_channel.get(rec.channel_id)
                        if dst_chan is None:
                            # Message references a channel we haven't seen
                            # (file is internally inconsistent). Skip rather
                            # than crash the whole recovery.
                            continue
                        writer.add_message(
                            channel_id=dst_chan,
                            log_time=rec.log_time,
                            publish_time=rec.publish_time,
                            data=rec.data,
                            sequence=rec.sequence,
                        )
                        n_msgs += 1
                        topic = src_id_to_topic[rec.channel_id]
                        topic_counts[topic] = topic_counts.get(topic, 0) + 1
                        if first_log_ns is None or rec.log_time < first_log_ns:
                            first_log_ns = rec.log_time
                        if last_log_ns is None or rec.log_time > last_log_ns:
                            last_log_ns = rec.log_time
                    # Anything else (Chunk-when-emit_chunks, ChunkIndex,
                    # Statistics, Attachment, etc.) — the writer will
                    # synthesize fresh summary records on finish().
            except McapError as e:
                # First parse error = truncation point. Everything before
                # it is already in the destination writer.
                print(
                    f"truncation detected at message {n_msgs}: {e}",
                    file=sys.stderr,
                )
        writer.finish()
    finally:
        dst_file.close()

    return n_msgs, topic_counts, first_log_ns, last_log_ns, spec_ids


def _refresh_sidecar(
    episode_dir: Path,
    mcap_path: Path,
    topic_counts: dict[str, int],
    first_log_ns: int | None,
    last_log_ns: int | None,
    spec_ids: set[tuple[str, str]],
) -> None:
    """Update the existing sidecar's derivable fields, mirroring Writer.close()."""
    sidecar = episode_dir / METADATA_FILE
    if not sidecar.exists():
        # No sidecar to refresh. Composable episodes legitimately ship
        # without one; the recovered mcap is self-describing.
        return
    try:
        meta = load_metadata(sidecar)
    except (ValueError, KeyError) as e:
        print(
            f"sidecar parse failed ({e}); skipping sidecar refresh",
            file=sys.stderr,
        )
        return

    checksum, size = compute_file_checksum_and_size(mcap_path)
    meta.data.uri = EPISODE_FILE
    meta.data.size_bytes = size
    meta.data.checksum_sha256 = checksum
    meta.data.topics = sorted(topic_counts)
    if first_log_ns is not None and last_log_ns is not None:
        meta.collection.duration_s = (last_log_ns - first_log_ns) / 1e9
    else:
        meta.collection.duration_s = 0.0
    meta.collection.length = max(topic_counts.values()) if topic_counts else 0
    if meta.robot.data_spec_id is None and len(spec_ids) == 1:
        sid, sver = next(iter(spec_ids))
        meta.robot.data_spec_id = f"{sid}_v{sver}"
    save_metadata(meta, sidecar)


if __name__ == "__main__":
    tyro.cli(main)
