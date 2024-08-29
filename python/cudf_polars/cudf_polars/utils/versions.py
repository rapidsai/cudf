# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Version utilities so that cudf_polars supports a range of polars versions."""

# ruff: noqa: SIM300
from __future__ import annotations

from packaging.version import parse

from polars import __version__

POLARS_VERSION = parse(__version__)

POLARS_VERSION_GE_10 = POLARS_VERSION >= parse("1.0")
POLARS_VERSION_GE_11 = POLARS_VERSION >= parse("1.1")
POLARS_VERSION_GE_12 = POLARS_VERSION >= parse("1.2")
POLARS_VERSION_GE_121 = POLARS_VERSION >= parse("1.2.1")
POLARS_VERSION_GT_10 = POLARS_VERSION > parse("1.0")
POLARS_VERSION_GT_11 = POLARS_VERSION > parse("1.1")
POLARS_VERSION_GT_12 = POLARS_VERSION > parse("1.2")

POLARS_VERSION_LE_12 = POLARS_VERSION <= parse("1.2")
POLARS_VERSION_LE_11 = POLARS_VERSION <= parse("1.1")
POLARS_VERSION_LT_12 = POLARS_VERSION < parse("1.2")
POLARS_VERSION_LT_11 = POLARS_VERSION < parse("1.1")

if POLARS_VERSION < parse("1.0"):  # pragma: no cover
    raise ImportError("cudf_polars requires py-polars v1.0 or greater.")
