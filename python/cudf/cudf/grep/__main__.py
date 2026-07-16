# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
"""Enable ``python -m cudf.grep``."""

from __future__ import annotations

from cudf.grep._grep import main

if __name__ == "__main__":
    raise SystemExit(main())
