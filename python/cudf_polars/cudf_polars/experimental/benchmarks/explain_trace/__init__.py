# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""
Convert benchmark trace JSONL output to explain-like tree output.

Usage (command-line):
    python -m cudf_polars.experimental.benchmarks.explain_trace pdsh_results.jsonl [--query Q]

Usage (programmatic):
    from cudf_polars.experimental.benchmarks.explain_trace import QueryPlan, load_jsonl

    records = load_jsonl("pdsh_results.jsonl")
    traces = records[0]["records"]["16"][0]["traces"]
    plan = QueryPlan.from_traces(traces)
    print(plan.render())
"""

from __future__ import annotations

from cudf_polars.experimental.benchmarks.explain_trace._core import (
    NodeStats,
    QueryPlan,
    get_traces_for_query,
    load_jsonl,
    main,
)

__all__ = [
    "NodeStats",
    "QueryPlan",
    "get_traces_for_query",
    "load_jsonl",
    "main",
]
