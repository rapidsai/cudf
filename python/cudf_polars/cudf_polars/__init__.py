# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""
An executor for polars logical plans.

This package implements an executor for polars logical plans using
pylibcudf to execute the plans on device.
"""

from __future__ import annotations

from cudf_polars._version import __git_commit__, __version__
from cudf_polars.callback import execute_with_cudf
from cudf_polars.dsl.translate import translate_ir

# Check we have a supported polars version
from cudf_polars.utils.versions import _ensure_polars_version

_ensure_polars_version()
del _ensure_polars_version

__all__: list[str] = [
    "execute_with_cudf",
    "translate_ir",
    "__git_commit__",
    "__version__",
]
