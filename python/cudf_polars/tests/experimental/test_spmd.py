# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Tests for SPMD execution mode."""

from __future__ import annotations

import pytest
from rapidsmpf.bootstrap import is_running_with_rrun
from rapidsmpf.rmm_resource_adaptor import RmmResourceAdaptor

import polars as pl

import rmm.mr

from cudf_polars.experimental.rapidsmpf.collectives.common import reserve_op_id
from cudf_polars.experimental.rapidsmpf.frontend.spmd import (
    SPMDEngine,
    allgather_polars_dataframe,
)

pytestmark = pytest.mark.spmd


def test_yields_context_and_engine() -> None:
    """SPMDEngine has comm and context properties."""
    with SPMDEngine() as engine:
        assert engine.comm is not None
        assert engine.context is not None
        assert isinstance(engine, pl.GPUEngine)


def test_single_communicator_outside_rrun() -> None:
    """Outside rrun the communicator has exactly one rank."""
    if is_running_with_rrun():
        pytest.skip("single-rank check only applies outside rrun")
    with SPMDEngine() as engine:
        assert engine.nranks == 1
        assert engine.rank == 0


def test_reserved_keys() -> None:
    """executor_options rejects reserved keys."""
    for key in ("runtime", "cluster", "spmd_context"):
        with (
            pytest.raises(TypeError, match="reserved"),
            SPMDEngine(executor_options={key: "anything"}),
        ):
            pass


def test_engine_options_reserved_keys() -> None:
    """engine_options rejects keys that are set explicitly by SPMDEngine."""
    for key in ("memory_resource", "executor"):
        with (
            pytest.raises(TypeError, match="reserved"),
            SPMDEngine(engine_options={key: "anything"}),
        ):
            pass


def test_engine_options_parquet_options() -> None:
    """engine_options forwards parquet_options to GPUEngine without error."""
    with SPMDEngine(engine_options={"parquet_options": {}}) as engine:
        assert isinstance(engine, pl.GPUEngine)


def test_scan() -> None:
    """Each rank scans its own single-row LazyFrame and gets that row back."""
    with SPMDEngine() as engine:
        lf = pl.LazyFrame({"a": [engine.rank], "b": [engine.rank * 10]})
        result = lf.collect(engine=engine)
        assert result.shape == (1, 2)
        assert result["a"].to_list() == [engine.rank]
        assert result["b"].to_list() == [engine.rank * 10]


def test_basic_query() -> None:
    """A simple in-memory LazyFrame can be collected."""
    with SPMDEngine() as engine:
        result = pl.LazyFrame({"a": [1, 2, 3], "b": [4, 5, 6]}).collect(engine=engine)
    assert result.shape == (3, 2)
    assert result["a"].to_list() == [1, 2, 3]


def test_collect_then_lazy_equivalent() -> None:
    """collect().lazy() preserves SPMD semantics: an intermediate materialize is a no-op.

    In SPMD mode a DataFrame is always rank-local.  When it is wrapped back
    into a LazyFrame the engine processes that rank's copy in full rather than
    re-slicing it across ranks.  So ``lf.collect().lazy().op.collect()`` must
    produce the same result as ``lf.op.collect()``.
    """
    with SPMDEngine() as engine:
        lf = pl.LazyFrame(
            {"a": [engine.rank, engine.rank + 1, engine.rank + 2], "b": [0, 1, 2]}
        )

        # One-step
        one_step = lf.filter(pl.col("b") >= 1).collect(engine=engine)

        # Two-step: materialize then re-wrap
        intermediate = lf.collect(engine=engine)
        two_step = intermediate.lazy().filter(pl.col("b") >= 1).collect(engine=engine)

    assert one_step.sort("a").equals(two_step.sort("a"))


def test_group_by() -> None:
    """Group-by on rank-local data, then allgather to verify the global result."""
    with SPMDEngine() as engine:
        lf = pl.LazyFrame({"a": [engine.rank], "b": [engine.rank * 10]})
        local_result = lf.group_by("a").agg(pl.col("b").sum()).collect(engine=engine)
        with reserve_op_id() as op_id:
            global_result = allgather_polars_dataframe(
                engine=engine, local_df=local_result, op_id=op_id
            )
        assert global_result.shape == (engine.nranks, 2)
        assert global_result.sort("a")["a"].to_list() == list(range(engine.nranks))
        assert global_result.sort("a")["b"].to_list() == [
            r * 10 for r in range(engine.nranks)
        ]


def test_allgather_polars_dataframe() -> None:
    """allgather_polars_dataframe collects every rank's contribution in rank order."""
    with SPMDEngine() as engine:
        local = pl.DataFrame({"rank": [engine.rank], "val": [engine.rank * 2]})
        with reserve_op_id() as op_id:
            result = allgather_polars_dataframe(
                engine=engine, local_df=local, op_id=op_id
            )
        assert result.shape == (engine.nranks, 2)
        assert result["rank"].to_list() == list(range(engine.nranks))
        assert result["val"].to_list() == [r * 2 for r in range(engine.nranks)]


def test_max_workers() -> None:
    """executor_options forwards rapidsmpf_py_executor_max_workers to the thread pool."""
    with SPMDEngine(
        executor_options={"rapidsmpf_py_executor_max_workers": 2}
    ) as engine:
        result = pl.LazyFrame({"a": [1, 2, 3]}).collect(engine=engine)
    assert result.shape == (3, 1)


def test_allgather_polars_dataframe_empty() -> None:
    """allgather handles an empty (zero-row) local DataFrame on every rank."""
    with SPMDEngine() as engine:
        local = pl.DataFrame(
            {"a": pl.Series([], dtype=pl.Int32), "b": pl.Series([], dtype=pl.Float64)}
        )
        with reserve_op_id() as op_id:
            result = allgather_polars_dataframe(
                engine=engine, local_df=local, op_id=op_id
            )
    assert result.shape == (0, 2)
    assert result.columns == ["a", "b"]
    assert result.dtypes == [pl.Int32, pl.Float64]


def test_mr_wrapped_as_current_inside_context() -> None:
    """Inside SPMDEngine the current device resource is RmmResourceAdaptor."""
    with SPMDEngine():
        assert isinstance(rmm.mr.get_current_device_resource(), RmmResourceAdaptor)


def test_mr_restored_after_context() -> None:
    """After SPMDEngine exits the original device resource is restored."""
    original = rmm.mr.get_current_device_resource()
    with SPMDEngine():
        pass
    assert rmm.mr.get_current_device_resource() is original


def test_allgather_polars_dataframe_multi_column() -> None:
    """allgather preserves column names, count, and dtypes for multi-column DataFrames."""
    with SPMDEngine() as engine:
        local = pl.DataFrame(
            {
                "rank": [engine.rank],
                "x": [float(engine.rank)],
                "label": [f"r{engine.rank}"],
            }
        )
        with reserve_op_id() as op_id:
            result = allgather_polars_dataframe(
                engine=engine, local_df=local, op_id=op_id
            )
        assert result.shape == (engine.nranks, 3)
        assert result.columns == ["rank", "x", "label"]
        sorted_result = result.sort("rank")
        assert sorted_result["rank"].to_list() == list(range(engine.nranks))
        assert sorted_result["x"].to_list() == [float(r) for r in range(engine.nranks)]
        assert sorted_result["label"].to_list() == [
            f"r{r}" for r in range(engine.nranks)
        ]
