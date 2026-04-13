# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Version utilities so that cudf_polars supports a range of polars versions."""

# ruff: noqa: SIM300
from __future__ import annotations

from packaging.version import parse

from polars import __version__

POLARS_VERSION = parse(__version__)
POLARS_LOWER_BOUND = parse("1.35")

POLARS_VERSION_LT_135 = POLARS_VERSION < parse("1.35")
POLARS_VERSION_LT_136 = POLARS_VERSION < parse("1.36")
POLARS_VERSION_LT_137 = POLARS_VERSION < parse("1.37")
POLARS_VERSION_LT_138 = POLARS_VERSION < parse("1.38")
POLARS_VERSION_LT_139 = POLARS_VERSION < parse("1.39")


def _ensure_polars_version() -> None:
    if POLARS_VERSION < POLARS_LOWER_BOUND:
        raise ImportError(
            f"cudf_polars requires py-polars v{POLARS_LOWER_BOUND} or greater."
        )  # pragma: no cover
