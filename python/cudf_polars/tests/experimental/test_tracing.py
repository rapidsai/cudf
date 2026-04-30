# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Integration tests for structlog tracing with rapidsmpf."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest


def test_structlog_streaming_node_events():
    """Test that structlog emits 'Streaming Actor' events when tracing is enabled."""
    pytest.importorskip("structlog")
    code = textwrap.dedent("""\
    import rmm
    import polars as pl

    rmm.mr.set_current_device_resource(rmm.mr.ManagedMemoryResource())
    from cudf_polars.experimental.rapidsmpf.frontend.spmd import SPMDEngine

    df = pl.DataFrame({"x": range(100), "y": ["a", "b"] * 50})
    q = df.lazy().filter(pl.col("x") > 50).group_by("y").agg(pl.col("x").sum())
    with SPMDEngine(executor_options={"max_rows_per_partition": 10}) as engine:
        q.collect(engine=engine)
    """)

    env = os.environ.copy()
    env["CUDF_POLARS_LOG_TRACES"] = "1"

    with subprocess.Popen(
        [sys.executable, "-c", code],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ) as proc:
        result, _ = proc.communicate(timeout=120)

    assert b"Streaming Actor" in result
    assert b"scope=actor" in result or b"'scope': 'actor'" in result
    assert b"actor_ir_id=" in result
    assert b"actor_ir_type=" in result
    assert b"chunk_count=" in result


def test_structlog_contains_expected_ir_types():
    """Test that structlog output contains expected IR types for a query."""
    pytest.importorskip("structlog")
    code = textwrap.dedent("""\
    import rmm
    import polars as pl

    rmm.mr.set_current_device_resource(rmm.mr.ManagedMemoryResource())
    from cudf_polars.experimental.rapidsmpf.frontend.spmd import SPMDEngine

    df = pl.DataFrame({"x": range(100), "y": ["a", "b"] * 50})
    q = df.lazy().filter(pl.col("x") > 50).group_by("y").agg(pl.col("x").sum())
    with SPMDEngine(executor_options={"max_rows_per_partition": 10}) as engine:
        q.collect(engine=engine)
    """)

    env = os.environ.copy()
    env["CUDF_POLARS_LOG_TRACES"] = "1"

    with subprocess.Popen(
        [sys.executable, "-c", code],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ) as proc:
        result, _ = proc.communicate(timeout=120)

    assert b"ir_type=DataFrameScan" in result
    assert b"ir_type=Filter" in result
    assert b"ir_type=GroupBy" in result


def test_structlog_disabled_by_default():
    """Test that structlog does NOT emit events when CUDF_POLARS_LOG_TRACES is not set."""
    pytest.importorskip("structlog")
    code = textwrap.dedent("""\
    import rmm
    import polars as pl

    rmm.mr.set_current_device_resource(rmm.mr.ManagedMemoryResource())
    from cudf_polars.experimental.rapidsmpf.frontend.spmd import SPMDEngine

    df = pl.DataFrame({"x": range(10), "y": ["a", "b"] * 5})
    q = df.lazy().filter(pl.col("x") > 5)
    with SPMDEngine(executor_options={"max_rows_per_partition": 5}) as engine:
        q.collect(engine=engine)
    """)

    env = os.environ.copy()
    env.pop("CUDF_POLARS_LOG_TRACES", None)

    with subprocess.Popen(
        [sys.executable, "-c", code],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ) as proc:
        result, _ = proc.communicate(timeout=120)

    assert b"Streaming Actor" not in result
