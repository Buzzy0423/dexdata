# Copyright (C) 2026 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Episode-directory discovery for the composable layer.

A composable episode lives in a directory whose direct contents include
``episode.mcap`` (matching :data:`writer.EPISODE_FILE`). The legacy and
composable layouts share this convention, so a tree containing both kinds
of episodes flatly enumerates with one walk.
"""

from __future__ import annotations

from pathlib import Path

from .writer import EPISODE_FILE


def discover_episodes(root: Path | str) -> list[Path]:
    """Return every directory under ``root`` that contains ``episode.mcap``.

    Sorted lexicographically for determinism.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"episode root not found: {root}")
    return sorted(p.parent for p in root.rglob(EPISODE_FILE))
