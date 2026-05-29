# Copyright (C) 2026 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Example: load a composable MCAP episode and inspect every signal.

Usage::

    python scripts/example_mcap_read.py --path <episode_dir>
"""

from __future__ import annotations

from pathlib import Path

import tyro

from dexdata import Reader


def main(path: Path) -> None:
    """Open a composable episode and print every signal's shape and dtype.

    Args:
        path: Path to the composable episode directory (contains
            ``episode.mcap`` and an optional ``metadata.json`` sidecar).
    """
    reader = Reader(path)
    episode = reader.read_episode()
    print(f"Loaded {len(episode.signals)} signals from {path}")
    if reader.metadata is not None:
        md = reader.metadata
        print(
            f"  Metadata: {md.collection.length} frames, "
            f"{md.collection.duration_s:.2f}s, "
            f"embodiment={md.robot.embodiment}, "
            f"task={md.task.task_id}"
        )
    for topic, arr in sorted(episode.signals.items()):
        print(f"  {topic}: shape={arr.shape} dtype={arr.dtype}")


if __name__ == "__main__":
    tyro.cli(main)
