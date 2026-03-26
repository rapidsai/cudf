# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""
Benchmark utilities - forwarding shim.

Dispatches to ``utils_new_frontends`` when ``--frontend`` appears in ``sys.argv``,
otherwise falls back to ``utils_legacy``.
"""

from __future__ import annotations

import sys


def _use_new_frontend() -> bool:
    # HACK: Inspect sys.argv to detect use of the new frontends
    # (e.g. ``--frontend ray``) without full argument parsing.
    # This only works when invoked from the CLI; direct imports always get the
    # legacy path. TODO: Remove this shim once the legacy path is deleted.
    return "--frontend" in sys.argv[1:]


if _use_new_frontend():
    from cudf_polars.experimental.benchmarks.utils_new_frontends import (
        COUNT_DTYPE,
        QueryResult,
        RunConfig,
        build_parser,
        get_data,
        parse_args,
        run_duckdb,
        run_polars,
    )
else:
    from cudf_polars.experimental.benchmarks.utils_legacy import (  # type: ignore[assignment]
        COUNT_DTYPE,
        QueryResult,
        RunConfig,
        build_parser,
        get_data,
        parse_args,
        run_duckdb,
        run_polars,
    )

__all__: list[str] = [
    "COUNT_DTYPE",
    "QueryResult",
    "RunConfig",
    "build_parser",
    "get_data",
    "parse_args",
    "run_duckdb",
    "run_polars",
]
