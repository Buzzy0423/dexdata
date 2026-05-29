# Copyright (C) 2026 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""CLI entry for ``python -m dexdata.composable.viz`` and ``dexdata-viz``.

Same surface as the legacy ``dexdata-viz`` it replaces (registered via
``project.scripts`` in ``pyproject.toml``):

* ``dexdata-viz <episode_dir>`` — open the Rerun viewer.
* ``dexdata-viz <episode_dir> --save --output X.rrd`` — write to file.
* ``dexdata-viz <X.rrd> --load`` — open a saved recording.
* ``dexdata-viz <episode_dir> --mode distant`` — serve web viewer.
* ``dexdata-viz <episode_dir> --lag --align {log,publish}`` — camera
  pipeline latency analysis.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Literal

import tyro

from ._episode import visualize_episode_dir
from ._lag import visualize_camera_lag


def cli(
    path: Path,
    mode: Literal["local", "distant"] = "local",
    save: bool = False,
    output: Path | None = None,
    load: bool = False,
    lag: bool = False,
    align: Literal["log", "publish"] = "log",
) -> None:
    """Visualize composable MCAP episodes using Rerun.

    Args:
        path: Episode directory, or a ``.rrd`` file when ``--load`` is set.
        mode: ``local`` spawns a local viewer; ``distant`` runs a server.
        save: Write the visualization to a ``.rrd`` file.
        output: Output path for ``--save`` (defaults to
            ``<episode_dir>/visualization.rrd``).
        load: Treat ``path`` as a saved ``.rrd`` file to open.
        lag: Visualize camera-pipeline latency instead of episode data.
        align: Timestamp source for ``--lag``: ``log`` or ``publish``.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if load:
        subprocess.Popen(
            ["rerun", str(path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return

    if lag:
        visualize_camera_lag(
            episode_dir=path,
            mode=mode,
            save=save,
            output_path=output,
            align=align,
        )
        return

    if save and output is None:
        output = path / "visualization.rrd"

    visualize_episode_dir(
        path,
        mode=mode,
        save=save,
        output_path=output,
    )


def main() -> None:
    tyro.cli(cli)


if __name__ == "__main__":
    main()
