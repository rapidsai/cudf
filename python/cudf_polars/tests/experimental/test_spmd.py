# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Tests for SPMD execution mode."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
from rapidsmpf.bootstrap import is_running_with_rrun
from rapidsmpf.rmm_resource_adaptor import RmmResourceAdaptor

import polars as pl

import rmm.mr

from cudf_polars.experimental.rapidsmpf.collectives.common import reserve_op_id
from cudf_polars.experimental.rapidsmpf.frontend.options import StreamingOptions
from cudf_polars.experimental.rapidsmpf.frontend.spmd import (
    SPMDEngine,
    allgather_polars_dataframe,
)
from cudf_polars.utils.config import MemoryResourceConfig

if TYPE_CHECKING:
    from rapidsmpf.communicator.communicator import Communicator

pytestmark = pytest.mark.spmd


def test_yields_context_and_engine(spmd_comm: Communicator) -> None:
    """SPMDEngine has comm and context properties."""
    with SPMDEngine(comm=spmd_comm) as engine:
        assert engine.comm is not None
        assert engine.context is not None
        assert isinstance(engine, pl.GPUEngine)


def test_from_options() -> None:
    """from_options with default StreamingOptions creates a valid SPMDEngine."""
    opts = StreamingOptions(fallback_mode="silent", raise_on_fail=True)
    with SPMDEngine.from_options(opts) as engine:
        assert engine.nranks >= 1


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


def test_engine_options_parquet_options(spmd_comm: Communicator) -> None:
    """engine_options forwards parquet_options to GPUEngine without error."""
    with SPMDEngine(comm=spmd_comm, engine_options={"parquet_options": {}}) as engine:
        assert isinstance(engine, pl.GPUEngine)


def test_scan(spmd_comm: Communicator) -> None:
    """Each rank scans its own single-row LazyFrame and gets that row back."""
    with SPMDEngine(comm=spmd_comm) as engine:
        lf = pl.LazyFrame({"a": [engine.rank], "b": [engine.rank * 10]})
        result = lf.collect(engine=engine)
        assert result.shape == (1, 2)
        assert result["a"].to_list() == [engine.rank]
        assert result["b"].to_list() == [engine.rank * 10]


def test_basic_query(spmd_comm: Communicator) -> None:
    """A simple in-memory LazyFrame can be collected."""
    with SPMDEngine(comm=spmd_comm) as engine:
        result = pl.LazyFrame({"a": [1, 2, 3], "b": [4, 5, 6]}).collect(engine=engine)
    assert result.shape == (3, 2)
    assert result["a"].to_list() == [1, 2, 3]


def test_collect_then_lazy_equivalent(spmd_comm: Communicator) -> None:
    """collect().lazy() preserves SPMD semantics: an intermediate materialize is a no-op.

    In SPMD mode a DataFrame is always rank-local.  When it is wrapped back
    into a LazyFrame the engine processes that rank's copy in full rather than
    re-slicing it across ranks.  So ``lf.collect().lazy().op.collect()`` must
    produce the same result as ``lf.op.collect()``.
    """
    with SPMDEngine(comm=spmd_comm) as engine:
        lf = pl.LazyFrame(
            {"a": [engine.rank, engine.rank + 1, engine.rank + 2], "b": [0, 1, 2]}
        )

        # One-step
        one_step = lf.filter(pl.col("b") >= 1).collect(engine=engine)

        # Two-step: materialize then re-wrap
        intermediate = lf.collect(engine=engine)
        two_step = intermediate.lazy().filter(pl.col("b") >= 1).collect(engine=engine)

    assert one_step.sort("a").equals(two_step.sort("a"))


def test_group_by(spmd_comm: Communicator) -> None:
    """Group-by on rank-local data, then allgather to verify the global result."""
    with SPMDEngine(comm=spmd_comm) as engine:
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


def test_allgather_polars_dataframe(spmd_comm: Communicator) -> None:
    """allgather_polars_dataframe collects every rank's contribution in rank order."""
    with SPMDEngine(comm=spmd_comm) as engine:
        local = pl.DataFrame({"rank": [engine.rank], "val": [engine.rank * 2]})
        with reserve_op_id() as op_id:
            result = allgather_polars_dataframe(
                engine=engine, local_df=local, op_id=op_id
            )
        assert result.shape == (engine.nranks, 2)
        assert result["rank"].to_list() == list(range(engine.nranks))
        assert result["val"].to_list() == [r * 2 for r in range(engine.nranks)]


def test_num_py_executors(spmd_comm: Communicator) -> None:
    """engine_options forwards num_py_executors to the thread pool."""
    with SPMDEngine(
        comm=spmd_comm,
        engine_options={"num_py_executors": 2},
    ) as engine:
        result = pl.LazyFrame({"a": [1, 2, 3]}).collect(engine=engine)
    assert result.shape == (3, 1)


def test_allgather_polars_dataframe_empty(spmd_comm: Communicator) -> None:
    """allgather handles an empty (zero-row) local DataFrame on every rank."""
    with SPMDEngine(comm=spmd_comm) as engine:
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


def test_mr_wrapped_as_current_inside_context(spmd_comm: Communicator) -> None:
    """Inside SPMDEngine the current device resource is RmmResourceAdaptor."""
    with SPMDEngine(comm=spmd_comm):
        assert isinstance(rmm.mr.get_current_device_resource(), RmmResourceAdaptor)


def test_mr_restored_after_context(spmd_comm: Communicator) -> None:
    """After SPMDEngine exits the original device resource is restored."""
    original = rmm.mr.get_current_device_resource()
    with SPMDEngine(comm=spmd_comm):
        pass
    assert rmm.mr.get_current_device_resource() is original


def test_allgather_polars_dataframe_multi_column(spmd_comm: Communicator) -> None:
    """allgather preserves column names, count, and dtypes for multi-column DataFrames."""
    with SPMDEngine(comm=spmd_comm) as engine:
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


# ---------------------------------------------------------------------------
# Tests specifically for the comm= argument
# ---------------------------------------------------------------------------


def test_comm_argument_reuses_communicator(spmd_comm: Communicator) -> None:
    """Passing comm= reuses the communicator across two engine lifetimes."""
    with SPMDEngine(comm=spmd_comm) as engine1:
        nranks = engine1.nranks
        rank = engine1.rank
    # engine1 is shut down; spmd_comm is still alive
    with SPMDEngine(comm=spmd_comm) as engine2:
        assert engine2.nranks == nranks
        assert engine2.rank == rank


def test_comm_not_closed_after_engine_shutdown(spmd_comm: Communicator) -> None:
    """The caller-provided comm survives engine.shutdown()."""
    with SPMDEngine(comm=spmd_comm):
        pass  # engine.shutdown() is called on __exit__
    # spmd_comm must still be accessible — not destroyed by engine teardown
    assert spmd_comm.rank >= 0


def test_comm_argument_mr_still_wrapped(spmd_comm: Communicator) -> None:
    """MR wrapping still happens even when comm is provided externally."""
    with SPMDEngine(comm=spmd_comm):
        assert isinstance(rmm.mr.get_current_device_resource(), RmmResourceAdaptor)


def test_comm_sequential_queries(spmd_comm: Communicator) -> None:
    """Two engines sharing a comm can each execute a query without interference."""
    with SPMDEngine(comm=spmd_comm) as engine:
        r1 = pl.LazyFrame({"a": [1, 2]}).collect(engine=engine)
    with SPMDEngine(comm=spmd_comm) as engine:
        r2 = pl.LazyFrame({"a": [3, 4]}).collect(engine=engine)
    assert r1["a"].to_list() == [1, 2]
    assert r2["a"].to_list() == [3, 4]


def test_shutdown_idempotent(spmd_comm: Communicator) -> None:
    """Calling shutdown() twice does not raise."""
    engine = SPMDEngine(comm=spmd_comm)
    engine.shutdown()
    engine.shutdown()


def test_memory_resource_config() -> None:
    """SPMDEngine uses the MR from memory_resource_config when provided."""
    config = MemoryResourceConfig(qualname="rmm.mr.CudaMemoryResource")
    opts = StreamingOptions(
        fallback_mode="silent",
        memory_resource_config=config,
    )
    with patch.object(
        MemoryResourceConfig,
        "create_memory_resource",
        wraps=config.create_memory_resource,
    ) as mock_create:
        with SPMDEngine.from_options(opts) as engine:
            assert engine.nranks >= 1
        mock_create.assert_called_once()


def test_comm_and_context_unavailable_after_shutdown(spmd_comm: Communicator) -> None:
    """Accessing comm or context after shutdown raises RuntimeError."""
    engine = SPMDEngine(comm=spmd_comm)
    engine.shutdown()
    with pytest.raises(RuntimeError, match="shutdown"):
        _ = engine.comm
    with pytest.raises(RuntimeError, match="shutdown"):
        _ = engine.context
