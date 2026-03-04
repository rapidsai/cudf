# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Integration tests for structlog tracing with rapidsmpf."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

from cudf_polars.testing.asserts import DEFAULT_CLUSTER, DEFAULT_RUNTIME


@pytest.mark.skipif(
    DEFAULT_RUNTIME != "rapidsmpf", reason="Requires 'rapidsmpf' runtime."
)
@pytest.mark.skipif(DEFAULT_CLUSTER != "single", reason="Requires 'single' cluster.")
def test_structlog_streaming_node_events():
    """Test that structlog emits 'Streaming Actor' events when tracing is enabled."""
    # Run in subprocess to control CUDF_POLARS_LOG_TRACES environment variable
    code = textwrap.dedent("""\
    import polars as pl
    import rmm

    df = pl.DataFrame({"x": range(100), "y": ["a", "b"] * 50})
    q = df.lazy().filter(pl.col("x") > 50).group_by("y").agg(pl.col("x").sum())
    engine = pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={
            "cluster": "single",
            "runtime": "rapidsmpf",
            "max_rows_per_partition": 10,
        },
        memory_resource=rmm.mr.ManagedMemoryResource(),
    )
    q.collect(engine=engine)
    """)

    # Build environment with tracing enabled
    env = os.environ.copy()
    env["CUDF_POLARS_LOG_TRACES"] = "1"

    result = subprocess.check_output(
        [sys.executable, "-c", code], env=env, stderr=subprocess.STDOUT
    )

    # Check for Streaming Actor events emitted by shutdown_on_error
    assert b"Streaming Actor" in result
    assert b"scope=actor" in result or b"'scope': 'actor'" in result
    assert b"actor_ir_id=" in result
    assert b"actor_ir_type=" in result
    assert b"chunk_count=" in result


@pytest.mark.skipif(
    DEFAULT_RUNTIME != "rapidsmpf", reason="Requires 'rapidsmpf' runtime."
)
@pytest.mark.skipif(DEFAULT_CLUSTER != "single", reason="Requires 'single' cluster.")
def test_structlog_contains_expected_ir_types():
    """Test that structlog output contains expected IR types for a query."""
    code = textwrap.dedent("""\
    import polars as pl
    import rmm

    df = pl.DataFrame({"x": range(100), "y": ["a", "b"] * 50})
    q = df.lazy().filter(pl.col("x") > 50).group_by("y").agg(pl.col("x").sum())
    engine = pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={
            "cluster": "single",
            "runtime": "rapidsmpf",
            "max_rows_per_partition": 10,
        },
        memory_resource=rmm.mr.ManagedMemoryResource(),
    )
    q.collect(engine=engine)
    """)

    env = os.environ.copy()
    env["CUDF_POLARS_LOG_TRACES"] = "1"

    result = subprocess.check_output(
        [sys.executable, "-c", code], env=env, stderr=subprocess.STDOUT
    )

    # Check for expected IR types in the query
    assert b"ir_type=DataFrameScan" in result
    assert b"ir_type=Filter" in result
    assert b"ir_type=GroupBy" in result
    assert b"ir_type=Repartition" in result


@pytest.mark.skipif(
    DEFAULT_RUNTIME != "rapidsmpf", reason="Requires 'rapidsmpf' runtime."
)
@pytest.mark.skipif(DEFAULT_CLUSTER != "single", reason="Requires 'single' cluster.")
def test_structlog_disabled_by_default():
    """Test that structlog does NOT emit events when CUDF_POLARS_LOG_TRACES is not set."""
    code = textwrap.dedent("""\
    import polars as pl
    import rmm

    df = pl.DataFrame({"x": range(10), "y": ["a", "b"] * 5})
    q = df.lazy().filter(pl.col("x") > 5)
    engine = pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={
            "cluster": "single",
            "runtime": "rapidsmpf",
            "max_rows_per_partition": 5,
        },
        memory_resource=rmm.mr.ManagedMemoryResource(),
    )
    q.collect(engine=engine)
    """)

    # Environment WITHOUT CUDF_POLARS_LOG_TRACES
    env = os.environ.copy()
    env.pop("CUDF_POLARS_LOG_TRACES", None)

    result = subprocess.check_output(
        [sys.executable, "-c", code], env=env, stderr=subprocess.STDOUT
    )

    # Should NOT see Streaming Actor events
    assert b"Streaming Actor" not in result
