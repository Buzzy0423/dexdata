# Copyright (C) 2026 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""CLI wrapper around :mod:`dexdata.viz`.

After ``pip install dexdata`` the same command is also available as the
``dexdata-viz`` console script and as
``python -m dexdata.viz``.
"""

from dexdata.viz.__main__ import main

if __name__ == "__main__":
    main()
