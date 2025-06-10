# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Version utilities so that cudf_polars supports a range of polars versions."""

# ruff: noqa: SIM300
from __future__ import annotations

from packaging.version import parse

from polars import __version__

POLARS_VERSION = parse(__version__)

POLARS_LOWER_BOUND = parse("1.25")
POLARS_VERSION_LT_128 = POLARS_VERSION < parse("1.28")
POLARS_VERSION_LT_129 = POLARS_VERSION < parse("1.29")
POLARS_VERSION_LT_130 = POLARS_VERSION < parse("1.30")


def _ensure_polars_version() -> None:
    if POLARS_VERSION < POLARS_LOWER_BOUND:
        raise ImportError(
            f"cudf_polars requires py-polars v{POLARS_LOWER_BOUND} or greater."
        )  # pragma: no cover
