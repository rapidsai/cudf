# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Version utilities so that cudf_polars supports a range of polars versions."""

# ruff: noqa: SIM300
from __future__ import annotations

from packaging.version import parse

from polars import __version__

POLARS_VERSION = parse(__version__)

POLARS_VERSION_LT_125 = POLARS_VERSION < parse("1.25")
POLARS_VERSION_LT_124 = POLARS_VERSION < parse("1.24")


def _ensure_polars_version() -> None:
    if POLARS_VERSION_LT_124:
        raise ImportError(
            "cudf_polars requires py-polars v1.24 or greater."
        )  # pragma: no cover
