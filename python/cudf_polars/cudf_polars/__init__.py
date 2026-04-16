# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""
An executor for polars logical plans.

This package implements an executor for polars logical plans using
pylibcudf to execute the plans on device.
"""

from __future__ import annotations

import contextlib
import io as _stdlib_io
import json as _stdlib_json
from typing import Any

import polars as pl

from cudf_polars._version import __git_commit__, __version__
from cudf_polars.callback import execute_with_cudf
from cudf_polars.dsl.translate import Translator

# Import hive helpers at module level so the import of 'cudf_polars.io' cannot
# shadow the stdlib 'io' alias above (importing a subpackage sets it as an
# attribute on the parent, which would overwrite a bare 'import io').
from cudf_polars.io.parquet_hive import _has_hive_scan, expand_hive_scan

# Check we have a supported polars version
from cudf_polars.utils.versions import _ensure_polars_version

_ensure_polars_version()
del _ensure_polars_version

# ---------------------------------------------------------------------------
# Transparent hive-partitioning expansion for GPU engine calls
#
# Polars raises NotImplementedError("scan with hive partitioning") from
# view_current_node() when the GPU callback encounters a hive-partitioned Scan.
# This is intentional: Polars signals that GPU engines must implement their own
# hive expansion rather than relying on the CPU path.
#
# We intercept LazyFrame.collect() and pre-expand hive scans into GPU-compatible
# plans *before* the NodeTraverser is created, bypassing the restriction.
#
# The hook is narrow: it only activates when all of the following are true:
#   1. engine is a pl.GPUEngine instance, and
#   2. the plan JSON contains a hive-partitioned Scan node.
# All other collect() calls pass through unmodified.
# ---------------------------------------------------------------------------

_original_lf_collect = pl.LazyFrame.collect


def _gpu_collect_with_hive_expansion(self: pl.LazyFrame, **kwargs: Any) -> pl.DataFrame:
    """Collect, expanding hive-partitioned scans for GPU engine compatibility."""
    engine = kwargs.get("engine")
    if isinstance(engine, pl.GPUEngine):
        try:
            buf = _stdlib_io.BytesIO()
            self._ldf.serialize_json(buf)
            buf.seek(0)
            plan_json = _stdlib_json.loads(buf.read())
        except Exception:
            plan_json = {}

        if _has_hive_scan(plan_json):
            with contextlib.suppress(Exception):
                self = expand_hive_scan(self)

    return _original_lf_collect(self, **kwargs)


pl.LazyFrame.collect = _gpu_collect_with_hive_expansion  # type: ignore[method-assign,assignment]

__all__: list[str] = [
    "Translator",
    "__git_commit__",
    "__version__",
    "execute_with_cudf",
]
