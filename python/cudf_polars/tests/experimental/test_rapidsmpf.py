# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.testing.asserts import (
    DEFAULT_CLUSTER,
    DEFAULT_RUNTIME,
    assert_gpu_result_equal,
)
from cudf_polars.testing.io import make_lazy_frame
from cudf_polars.utils.config import ConfigOptions

REQUIRE_TASKS_RUNTIME = pytest.mark.skipif(
    DEFAULT_RUNTIME != "tasks", reason="Requires 'tasks' runtime."
)


@REQUIRE_TASKS_RUNTIME
@pytest.mark.parametrize("rapidsmpf_spill", [False, True])
@pytest.mark.parametrize("max_rows_per_partition", [1, 5])
def test_join_rapidsmpf(
    max_rows_per_partition: int,
    rapidsmpf_spill: bool,  # noqa: FBT001
) -> None:
    # Check that we have a distributed cluster running.
    # This tests must be run with: --cluster='distributed'
    distributed = pytest.importorskip("distributed")
    try:
        client = distributed.get_client()
    except ValueError:
        pytest.skip(reason="Requires distributed execution.")

    # check that we have a rapidsmpf cluster running
    pytest.importorskip("rapidsmpf")
    try:
        # This will result in a ValueError if the
        # cluster isn't compatible with rapidsmpf.
        # Otherwise, it's a no-op if the cluster
        # was already bootstrapped.
        from rapidsmpf.integrations.dask import bootstrap_dask_cluster

        bootstrap_dask_cluster(client)
    except ValueError:
        pytest.skip(reason="Requires a rapidsmpf-bootstrapped cluster.")

    # Setup the GPUEngine config
    engine = pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={
            "max_rows_per_partition": max_rows_per_partition,
            "broadcast_join_limit": 2,
            "shuffle_method": "rapidsmpf",
            "cluster": "distributed",
            "runtime": DEFAULT_RUNTIME,
            "rapidsmpf_spill": rapidsmpf_spill,
        },
    )

    left = pl.LazyFrame(
        {
            "x": range(15),
            "y": [1, 2, 3] * 5,
            "z": [1.0, 2.0, 3.0, 4.0, 5.0] * 3,
        }
    )
    right = pl.LazyFrame(
        {
            "xx": range(6),
            "y": [2, 4, 3] * 2,
            "zz": [1, 2] * 3,
        }
    )
    q = left.join(right, on="y", how="inner")

    assert_gpu_result_equal(q, engine=engine, check_row_order=False)


@REQUIRE_TASKS_RUNTIME
@pytest.mark.parametrize("max_rows_per_partition", [1, 5])
def test_join_rapidsmpf_single(max_rows_per_partition: int) -> None:
    # check that we have a rapidsmpf cluster running
    pytest.importorskip("rapidsmpf")

    # Setup the GPUEngine config
    engine = pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={
            "max_rows_per_partition": max_rows_per_partition,
            "broadcast_join_limit": 2,
            "shuffle_method": "rapidsmpf",
            "cluster": "single",
            "runtime": DEFAULT_RUNTIME,
        },
    )

    left = pl.LazyFrame(
        {
            "x": range(15),
            "y": [1, 2, 3] * 5,
            "z": [1.0, 2.0, 3.0, 4.0, 5.0] * 3,
        }
    )
    right = pl.LazyFrame(
        {
            "xx": range(6),
            "y": [2, 4, 3] * 2,
            "zz": [1, 2] * 3,
        }
    )
    q = left.join(right, on="y", how="inner")

    assert_gpu_result_equal(q, engine=engine, check_row_order=False)


@REQUIRE_TASKS_RUNTIME
def test_join_rapidsmpf_single_private_config() -> None:
    # The user may not specify "rapidsmpf-single" directly
    engine = pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={
            "shuffle_method": "rapidsmpf-single",
            "cluster": "single",
            "runtime": DEFAULT_RUNTIME,
        },
    )
    with pytest.raises(ValueError, match="not a supported shuffle method"):
        ConfigOptions.from_polars_engine(engine)


def test_rapidsmpf_spill_single_unsupported() -> None:
    # check that we have a rapidsmpf cluster running
    pytest.importorskip("rapidsmpf")

    # rapidsmpf_spill=True is not yet supported with single-GPU cluster.
    engine = pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={
            "shuffle_method": "rapidsmpf",
            "cluster": "single",
            "runtime": DEFAULT_RUNTIME,
            "rapidsmpf_spill": True,
        },
    )
    with pytest.raises(ValueError, match="rapidsmpf_spill.*not supported.*single"):
        ConfigOptions.from_polars_engine(engine)


@REQUIRE_TASKS_RUNTIME
@pytest.mark.parametrize("max_rows_per_partition", [1, 5])
def test_sort_rapidsmpf(max_rows_per_partition: int) -> None:
    # Require rapidsmpf, but don't require a distributed cluster,
    # because single-worker shuffle can be used.
    pytest.importorskip("rapidsmpf")

    # Setup the GPUEngine config
    engine = pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={
            "max_rows_per_partition": max_rows_per_partition,
            "shuffle_method": "rapidsmpf",
            "cluster": DEFAULT_CLUSTER,
            "runtime": DEFAULT_RUNTIME,
        },
    )

    df = pl.LazyFrame(
        {
            "x": range(15),
            "y": [1, 2, 3] * 5,
            "z": [1.0, 2.0, 3.0, 4.0, 5.0] * 3,
        }
    )
    q = df.sort(by=["y", "z"])

    assert_gpu_result_equal(q, engine=engine, check_row_order=True)


@REQUIRE_TASKS_RUNTIME
def test_sort_stable_rapidsmpf_warns():
    pytest.importorskip("rapidsmpf")

    engine = pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={
            "max_rows_per_partition": 3,
            "cluster": DEFAULT_CLUSTER,
            "runtime": DEFAULT_RUNTIME,
            "shuffle_method": "rapidsmpf",
            "fallback_mode": "warn",
        },
    )

    df = pl.LazyFrame(
        {
            "x": range(15),
            "y": [1, 2, 3] * 5,
            "z": [1.0, 2.0, 3.0, 4.0, 5.0] * 3,
        }
    )

    q = df.sort(by=["y", "z"], maintain_order=True)
    with pytest.warns(UserWarning, match="Falling back to shuffle_method='tasks'."):
        assert_gpu_result_equal(q, engine=engine, check_row_order=True)


@pytest.mark.parametrize("source_format", ["frame", "parquet", "csv"])
def test_simple_query_with_distributed_support(tmp_path, source_format) -> None:
    # Test a trivial query that works for both the
    # "tasks" and "rapidsmpf" runtimes in distributed mode.
    # Note: For "frame" (DataFrameScan), the DataFrameScan class overrides
    # the base Node.__reduce__ method to serialize the polars DataFrame
    # from _non_child_args instead of the non-picklable PyDataFrame.

    # Check that we have a distributed cluster running.
    # This tests must be run with: --cluster='distributed'
    distributed = pytest.importorskip("distributed")
    try:
        client = distributed.get_client()
    except ValueError:
        pytest.skip(reason="Requires distributed execution.")

    # check that we have a rapidsmpf cluster running
    pytest.importorskip("rapidsmpf")
    try:
        from rapidsmpf.integrations.dask import bootstrap_dask_cluster

        bootstrap_dask_cluster(client)
    except ValueError:
        pytest.skip(reason="Requires a rapidsmpf-bootstrapped cluster.")

    # Setup the GPUEngine config
    engine = pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={
            "max_rows_per_partition": 2,
            "cluster": "distributed",
            "runtime": DEFAULT_RUNTIME,
        },
    )

    # Create a simple DataFrame
    df = pl.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
            "b": [10, 20, 30, 40, 50],
        }
    )

    # Create LazyFrame based on source format
    if source_format == "frame":
        lf = make_lazy_frame(df, fmt="frame")
    else:
        lf = make_lazy_frame(df, fmt=source_format, path=tmp_path, n_files=2)

    # Simple query: select and filter
    q = lf.select("a", "b").filter(pl.col("a") > 2)

    # Should warn about distributed execution being under construction (if distributed)
    if DEFAULT_RUNTIME == "rapidsmpf":
        with pytest.warns(UserWarning, match="UNDER CONSTRUCTION"):
            result = q.collect(engine=engine)
    else:
        result = q.collect(engine=engine)

    # Check the result is correct
    expected = df.lazy().select("a", "b").filter(pl.col("a") > 2).collect()
    assert result.equals(expected)
