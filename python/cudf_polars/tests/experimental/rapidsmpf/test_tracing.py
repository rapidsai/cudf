# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Integration tests for runtime tracing with rapidsmpf."""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

import polars as pl

from cudf_polars.testing.asserts import DEFAULT_CLUSTER, DEFAULT_RUNTIME
from cudf_polars.testing.io import make_partitioned_source


def get_engine(output_path: str, parquet_options: dict | None = None) -> pl.GPUEngine:
    return pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={
            "cluster": DEFAULT_CLUSTER,
            "runtime": DEFAULT_RUNTIME,
            "tracing": {"output_path": str(output_path)},
            "max_rows_per_partition": 10,
            "target_partition_size": 500,
        },
        parquet_options=parquet_options,
    )


@pytest.fixture
def df():
    return pl.DataFrame({"x": range(100), "y": ["a", "b"] * 50})


@pytest.mark.skipif(DEFAULT_CLUSTER != "single", reason="Requires 'single' cluster.")
def test_tracing_basic_query(tmp_path, df):
    """Test tracing output with a DataFrameScan query."""
    output_path = tmp_path / "dataframe_scan_trace.txt"
    engine = get_engine(output_path)
    q = df.lazy().filter(pl.col("x") > 50).group_by("y").agg(pl.col("x").sum())
    q.collect(engine=engine)
    content = output_path.read_text()
    assert "GROUPBY ('y',) ('y', 'x') rows=2 chunks=1" in content
    assert "REPARTITION ('y', 'x') rows=10 chunks=1" in content
    assert "FILTER ('x', 'y') rows=49 chunks=10" in content
    assert "DATAFRAMESCAN ('x', 'y') rows=100 chunks=10" in content


@pytest.mark.skipif(DEFAULT_CLUSTER != "single", reason="Requires 'single' cluster.")
def test_tracing_scan_parquet_python(tmp_path, df):
    """Test tracing output with a ScanParquet query."""
    output_path = tmp_path / "scan_parquet_trace.txt"
    engine = get_engine(output_path, parquet_options={"use_rapidsmpf_native": False})
    pq_path = tmp_path / "pq_python"
    pq_path.mkdir()
    make_partitioned_source(df, pq_path, "parquet", n_files=5)
    q = pl.scan_parquet(pq_path).filter(pl.col("x") > 50).select(["x", "y"])
    q.collect(engine=engine)
    content = output_path.read_text()
    assert "SCAN PARQUET ('x', 'y') rows=49 chunks=5" in content


@pytest.mark.skipif(DEFAULT_CLUSTER != "single", reason="Requires 'single' cluster.")
def test_tracing_scan_parquet_native(tmp_path, df):
    """Test tracing output with a ScanParquet query."""
    output_path = tmp_path / "scan_parquet_trace.txt"
    engine = get_engine(output_path, parquet_options={"use_rapidsmpf_native": True})
    pq_path = tmp_path / "pq_native"
    pq_path.mkdir()
    make_partitioned_source(df, pq_path, "parquet", n_files=5)
    q = pl.scan_parquet(pq_path).filter(pl.col("x") > 50).select(["x", "y"])
    q.collect(engine=engine)
    content = output_path.read_text()
    # We can count the chunks but not the rows for "native" parquet
    # (row count is omitted when unavailable)
    assert "SCAN PARQUET ('x', 'y') chunks=5" in content


@pytest.mark.skipif(
    DEFAULT_RUNTIME != "rapidsmpf", reason="Requires 'rapidsmpf' runtime."
)
@pytest.mark.skipif(DEFAULT_CLUSTER != "single", reason="Requires 'single' cluster.")
def test_structlog_streaming_node_events():
    """Test that structlog emits 'Streaming Node' events when tracing is enabled."""
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

    env = {
        "CUDF_POLARS__EXECUTOR": "streaming",
        "CUDF_POLARS_LOG_TRACES": "1",
    }

    result = subprocess.check_output([sys.executable, "-c", code], env=env)

    # Check for Streaming Node events emitted by shutdown_on_error
    assert b"Streaming Node" in result
    assert b"ir_id=" in result
    assert b"ir_type=" in result
    assert b"chunks=" in result
