# Copyright (C) 2026 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Verify a composable dexdata episode (or a tree of them).

Default pass is *structural*: opens ``episode.mcap``, reads its summary,
cross-checks the ``metadata.json`` sidecar (size, sha256, topic set). Cheap
enough to run across a catalogue.

``--deep`` adds:

* full ``iter_messages()`` pass — catches mid-file truncation that the
  recorded summary's chunk index may still claim is reachable.
* :class:`dexdata.Reader` construction — confirms the spec rehydrates and
  every channel resolves to a handler.

Usage::

    python scripts/verify_episode.py --path <episode_dir_or_root>
    python scripts/verify_episode.py --path <root> --deep

Exit code is the number of failing episodes (capped at 255), so this is
shell-pipelineable.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import tyro
from mcap.exceptions import McapError
from mcap.reader import make_reader

from dexdata import EPISODE_FILE, Reader, discover_episodes
from dexdata.metadata import (
    METADATA_FILE,
    compute_file_checksum_and_size,
    load_metadata,
)

MCAP_MAGIC = b"\x89MCAP0\r\n"


@dataclass
class Report:
    """Per-episode result; ``problems`` empty → episode passed."""

    episode_dir: Path
    problems: list[str] = field(default_factory=list)
    # Soft notes (sidecar absent, fields not yet finalized) that don't
    # mark the episode bad but are worth surfacing.
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


def verify_episode(episode_dir: Path, *, deep: bool = False) -> Report:
    """Run checks against one episode directory."""
    rep = Report(episode_dir=episode_dir)
    mcap_path = episode_dir / EPISODE_FILE

    if not mcap_path.exists():
        rep.problems.append(f"missing {EPISODE_FILE}")
        return rep

    size = mcap_path.stat().st_size
    if size < len(MCAP_MAGIC) * 2:
        rep.problems.append(f"file too small to be MCAP ({size} bytes)")
        return rep

    with open(mcap_path, "rb") as f:
        if f.read(len(MCAP_MAGIC)) != MCAP_MAGIC:
            rep.problems.append("missing MCAP magic at file start")
            return rep

    # Summary section is written by ``McapWriter.finish()``. Its absence is
    # the canonical signal that the recorder crashed before close().
    try:
        with open(mcap_path, "rb") as f:
            reader = make_reader(f)
            summary = reader.get_summary()
    except McapError as e:
        msg = str(e)
        # An unclosed/truncated mcap has no footer record at end-of-file,
        # so the seeking reader interprets trailing bytes as a record with
        # bogus opcode/length. Surface that interpretation up front.
        if "opcode" in msg or "exceeds limit" in msg:
            rep.problems.append(
                f"mcap parse failed ({e}); likely truncated / unclosed file"
            )
        else:
            rep.problems.append(f"mcap parse failed: {e}")
        return rep

    if summary is None:
        rep.problems.append(
            "no summary section — recorder likely crashed before close()"
        )
        # Without a summary we can't trust further structural claims; bail.
        return rep

    mcap_topics = {ch.topic for ch in summary.channels.values()}
    if not mcap_topics:
        rep.problems.append("summary has zero channels")

    _check_sidecar(episode_dir, mcap_path, size, mcap_topics, rep)

    if deep:
        _deep_checks(mcap_path, rep)

    return rep


def _check_sidecar(
    episode_dir: Path,
    mcap_path: Path,
    file_size: int,
    mcap_topics: set[str],
    rep: Report,
) -> None:
    sidecar = episode_dir / METADATA_FILE
    if not sidecar.exists():
        rep.notes.append(f"no {METADATA_FILE} sidecar")
        return

    try:
        meta = load_metadata(sidecar)
    except (ValueError, KeyError) as e:
        rep.problems.append(f"sidecar parse failed: {e}")
        return

    # Writer initializes these to dataclass defaults up front and only
    # rewrites them on a clean close(). Either default is a tell that
    # close() never ran.
    if meta.data.size_bytes is None or meta.data.checksum_sha256 is None:
        rep.problems.append(
            "sidecar size/checksum not finalized — Writer.close() never ran"
        )
        return

    if meta.data.size_bytes != file_size:
        rep.problems.append(
            f"sidecar size_bytes={meta.data.size_bytes} != on-disk {file_size}"
        )

    checksum, size = compute_file_checksum_and_size(mcap_path)
    if checksum != meta.data.checksum_sha256:
        rep.problems.append(
            f"sha256 mismatch: file={checksum[:12]}… "
            f"sidecar={meta.data.checksum_sha256[:12]}…"
        )

    sidecar_topics = set(meta.data.topics)
    missing = sidecar_topics - mcap_topics
    extra = mcap_topics - sidecar_topics
    if missing:
        rep.problems.append(f"sidecar lists topics not in MCAP: {sorted(missing)}")
    if extra:
        rep.problems.append(f"MCAP has topics not in sidecar: {sorted(extra)}")


def _deep_checks(mcap_path: Path, rep: Report) -> None:
    # Walk every message — catches truncation past the summary offset, or
    # chunks that the summary claims but the body lost.
    try:
        with open(mcap_path, "rb") as f:
            reader = make_reader(f)
            count = 0
            for _schema, _channel, _msg in reader.iter_messages():
                count += 1
        rep.notes.append(f"iterated {count} messages")
    except McapError as e:
        rep.problems.append(f"iter_messages failed: {e}")
        return

    # Spec recovery + handler construction. Catches channel metadata that's
    # technically valid MCAP but unreadable as a composable episode.
    try:
        Reader(mcap_path.parent)
    except Exception as e:  # noqa: BLE001 — handler errors are diverse
        rep.problems.append(f"Reader() failed: {type(e).__name__}: {e}")


def _format_report(rep: Report) -> str:
    head = ("PASS" if rep.ok else "FAIL") + f"  {rep.episode_dir}"
    lines = [head]
    for p in rep.problems:
        lines.append(f"    ! {p}")
    for n in rep.notes:
        lines.append(f"    · {n}")
    return "\n".join(lines)


def main(path: Path, deep: bool = False) -> None:
    """Verify one episode dir or every episode under ``path``.

    Args:
        path: An episode directory (contains ``episode.mcap``) or a root
            directory to walk recursively.
        deep: Add a full message iterate + Reader round-trip per episode.
    """
    if (path / EPISODE_FILE).exists():
        episode_dirs = [path]
    else:
        episode_dirs = discover_episodes(path)
        if not episode_dirs:
            print(f"no episodes under {path}", file=sys.stderr)
            sys.exit(1)

    n_fail = 0
    for ep in episode_dirs:
        rep = verify_episode(ep, deep=deep)
        print(_format_report(rep))
        if not rep.ok:
            n_fail += 1

    total = len(episode_dirs)
    print(f"\n{total - n_fail}/{total} episodes ok ({n_fail} failed)")
    sys.exit(min(n_fail, 255))


if __name__ == "__main__":
    tyro.cli(main)
