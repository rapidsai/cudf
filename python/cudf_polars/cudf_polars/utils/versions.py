# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Version utilities so that cudf_polars supports a range of polars versions."""

# ruff: noqa: SIM300
from __future__ import annotations

from packaging.version import parse

from polars import __version__

POLARS_VERSION = parse(__version__)

POLARS_VERSION_GE_13 = POLARS_VERSION >= parse("1.3")
POLARS_VERSION_GT_13 = POLARS_VERSION > parse("1.3")
POLARS_VERSION_LT_13 = POLARS_VERSION < parse("1.3")

if POLARS_VERSION < parse("1.3"):
    raise ImportError(
        "cudf_polars requires py-polars v1.3 or greater."
    )  # pragma: no cover
