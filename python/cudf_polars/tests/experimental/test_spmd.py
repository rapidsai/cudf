# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Tests for SPMD execution mode."""

from __future__ import annotations

import os
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


@pytest.fixture
def comm(spmd_engine: SPMDEngine) -> Communicator:
    """Communicator from the shared :class:`SPMDEngine` for local construction.

    Most tests in this module need to construct their own
    :class:`SPMDEngine` to exercise lifecycle, construction-time
    options, MR-state semantics, or :meth:`SPMDEngine._reset`.
    """
    return spmd_engine.comm


def test_yields_context_and_engine(spmd_engine: SPMDEngine) -> None:
    """SPMDEngine has comm and context properties."""
    assert spmd_engine.comm is not None
    assert spmd_engine.context is not None
    assert isinstance(spmd_engine, pl.GPUEngine)


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


def test_engine_options_parquet_options(comm: Communicator) -> None:
    """engine_options forwards parquet_options to GPUEngine without error."""
    with SPMDEngine(comm=comm, engine_options={"parquet_options": {}}) as engine:
        assert isinstance(engine, pl.GPUEngine)


def test_scan(spmd_engine: SPMDEngine) -> None:
    """Each rank scans its own single-row LazyFrame and gets that row back."""
    lf = pl.LazyFrame({"a": [spmd_engine.rank], "b": [spmd_engine.rank * 10]})
    result = lf.collect(engine=spmd_engine)
    assert result.shape == (1, 2)
    assert result["a"].to_list() == [spmd_engine.rank]
    assert result["b"].to_list() == [spmd_engine.rank * 10]


def test_basic_query(spmd_engine: SPMDEngine) -> None:
    """A simple in-memory LazyFrame can be collected."""
    result = pl.LazyFrame({"a": [1, 2, 3], "b": [4, 5, 6]}).collect(engine=spmd_engine)
    assert result.shape == (3, 2)
    assert result["a"].to_list() == [1, 2, 3]


def test_collect_then_lazy_equivalent(spmd_engine: SPMDEngine) -> None:
    """collect().lazy() preserves SPMD semantics: an intermediate materialize is a no-op.

    In SPMD mode a DataFrame is always rank-local.  When it is wrapped back
    into a LazyFrame the engine processes that rank's copy in full rather than
    re-slicing it across ranks.  So ``lf.collect().lazy().op.collect()`` must
    produce the same result as ``lf.op.collect()``.
    """
    rank = spmd_engine.rank
    lf = pl.LazyFrame({"a": [rank, rank + 1, rank + 2], "b": [0, 1, 2]})

    # One-step
    one_step = lf.filter(pl.col("b") >= 1).collect(engine=spmd_engine)

    # Two-step: materialize then re-wrap
    intermediate = lf.collect(engine=spmd_engine)
    two_step = intermediate.lazy().filter(pl.col("b") >= 1).collect(engine=spmd_engine)

    assert one_step.sort("a").equals(two_step.sort("a"))


def test_group_by(spmd_engine: SPMDEngine) -> None:
    """Group-by on rank-local data, then allgather to verify the global result."""
    lf = pl.LazyFrame({"a": [spmd_engine.rank], "b": [spmd_engine.rank * 10]})
    local_result = lf.group_by("a").agg(pl.col("b").sum()).collect(engine=spmd_engine)
    with reserve_op_id() as op_id:
        global_result = allgather_polars_dataframe(
            engine=spmd_engine, local_df=local_result, op_id=op_id
        )
    assert global_result.shape == (spmd_engine.nranks, 2)
    assert global_result.sort("a")["a"].to_list() == list(range(spmd_engine.nranks))
    assert global_result.sort("a")["b"].to_list() == [
        r * 10 for r in range(spmd_engine.nranks)
    ]


def test_allgather_polars_dataframe(spmd_engine: SPMDEngine) -> None:
    """allgather_polars_dataframe collects every rank's contribution in rank order."""
    local = pl.DataFrame({"rank": [spmd_engine.rank], "val": [spmd_engine.rank * 2]})
    with reserve_op_id() as op_id:
        result = allgather_polars_dataframe(
            engine=spmd_engine, local_df=local, op_id=op_id
        )
    assert result.shape == (spmd_engine.nranks, 2)
    assert result["rank"].to_list() == list(range(spmd_engine.nranks))
    assert result["val"].to_list() == [r * 2 for r in range(spmd_engine.nranks)]


def test_num_py_executors(comm: Communicator) -> None:
    """executor_options forwards num_py_executors to the thread pool."""
    with SPMDEngine(
        comm=comm,
        executor_options={"num_py_executors": 2},
    ) as engine:
        result = pl.LazyFrame({"a": [1, 2, 3]}).collect(engine=engine)
    assert result.shape == (3, 1)


def test_allgather_polars_dataframe_empty(spmd_engine: SPMDEngine) -> None:
    """allgather handles an empty (zero-row) local DataFrame on every rank."""
    local = pl.DataFrame(
        {"a": pl.Series([], dtype=pl.Int32), "b": pl.Series([], dtype=pl.Float64)}
    )
    with reserve_op_id() as op_id:
        result = allgather_polars_dataframe(
            engine=spmd_engine, local_df=local, op_id=op_id
        )
    assert result.shape == (0, 2)
    assert result.columns == ["a", "b"]
    assert result.dtypes == [pl.Int32, pl.Float64]


def test_mr_wrapped_as_current_inside_context(comm: Communicator) -> None:
    """Inside SPMDEngine the current device resource is RmmResourceAdaptor."""
    with SPMDEngine(comm=comm):
        assert isinstance(rmm.mr.get_current_device_resource(), RmmResourceAdaptor)


def test_mr_restored_after_context(comm: Communicator) -> None:
    """After SPMDEngine exits the original device resource is restored."""
    original = rmm.mr.get_current_device_resource()
    with SPMDEngine(comm=comm):
        pass
    assert rmm.mr.get_current_device_resource() is original


def test_allgather_polars_dataframe_multi_column(spmd_engine: SPMDEngine) -> None:
    """allgather preserves column names, count, and dtypes for multi-column DataFrames."""
    local = pl.DataFrame(
        {
            "rank": [spmd_engine.rank],
            "x": [float(spmd_engine.rank)],
            "label": [f"r{spmd_engine.rank}"],
        }
    )
    with reserve_op_id() as op_id:
        result = allgather_polars_dataframe(
            engine=spmd_engine, local_df=local, op_id=op_id
        )
    assert result.shape == (spmd_engine.nranks, 3)
    assert result.columns == ["rank", "x", "label"]
    sorted_result = result.sort("rank")
    assert sorted_result["rank"].to_list() == list(range(spmd_engine.nranks))
    assert sorted_result["x"].to_list() == [float(r) for r in range(spmd_engine.nranks)]
    assert sorted_result["label"].to_list() == [
        f"r{r}" for r in range(spmd_engine.nranks)
    ]


# ---------------------------------------------------------------------------
# Tests specifically for the comm= argument
# ---------------------------------------------------------------------------


def test_comm_argument_reuses_communicator(comm: Communicator) -> None:
    """Passing comm= reuses the communicator across two engine lifetimes."""
    with SPMDEngine(comm=comm) as engine1:
        nranks = engine1.nranks
        rank = engine1.rank
    # engine1 is shut down; the shared comm is still alive
    with SPMDEngine(comm=comm) as engine2:
        assert engine2.nranks == nranks
        assert engine2.rank == rank


def test_comm_not_closed_after_engine_shutdown(comm: Communicator) -> None:
    """The caller-provided comm survives engine.shutdown()."""
    with SPMDEngine(comm=comm):
        pass  # engine.shutdown() is called on __exit__
    # comm must still be accessible — not destroyed by engine teardown
    assert comm.rank >= 0


def test_comm_argument_mr_still_wrapped(comm: Communicator) -> None:
    """MR wrapping still happens even when comm is provided externally."""
    with SPMDEngine(comm=comm):
        assert isinstance(rmm.mr.get_current_device_resource(), RmmResourceAdaptor)


def test_comm_sequential_queries(comm: Communicator) -> None:
    """Two engines sharing a comm can each execute a query without interference."""
    with SPMDEngine(comm=comm) as engine:
        r1 = pl.LazyFrame({"a": [1, 2]}).collect(engine=engine)
    with SPMDEngine(comm=comm) as engine:
        r2 = pl.LazyFrame({"a": [3, 4]}).collect(engine=engine)
    assert r1["a"].to_list() == [1, 2]
    assert r2["a"].to_list() == [3, 4]


def test_shutdown_idempotent(comm: Communicator) -> None:
    """Calling shutdown() twice does not raise."""
    engine = SPMDEngine(comm=comm)
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


def test_comm_and_context_unavailable_after_shutdown(comm: Communicator) -> None:
    """Accessing comm or context after shutdown raises RuntimeError."""
    engine = SPMDEngine(comm=comm)
    engine.shutdown()
    with pytest.raises(RuntimeError, match="shutdown"):
        _ = engine.comm
    with pytest.raises(RuntimeError, match="shutdown"):
        _ = engine.context


def test_run(spmd_engine: SPMDEngine) -> None:
    result = spmd_engine._run(os.getpid)
    assert result == [os.getpid()]


def test_reset_keeps_comm_alive(comm: Communicator) -> None:
    """``_reset`` must not rebuild the communicator."""
    with SPMDEngine(
        comm=comm, executor_options={"max_rows_per_partition": 10}
    ) as engine:
        comm_before = engine.comm
        engine._reset(executor_options={"max_rows_per_partition": 7})
        # Same Communicator instance — caller-provided comm is preserved.
        assert engine.comm is comm_before
        # Engine still drives a real query.
        result = pl.LazyFrame({"a": [1, 2, 3]}).collect(engine=engine)
        assert sorted(result["a"].to_list()) == [1, 2, 3]


def test_reset_updates_executor_options(comm: Communicator) -> None:
    """``_reset`` updates the polars-layer config to the new options."""
    from cudf_polars.utils.config import SPMDContext

    with SPMDEngine(
        comm=comm, executor_options={"max_rows_per_partition": 10}
    ) as engine:
        engine._reset(executor_options={"max_rows_per_partition": 42})

        opts = engine.config["executor_options"]
        assert opts["max_rows_per_partition"] == 42
        # Reserved keys are still injected by ``_reset``.
        assert opts["runtime"] == "rapidsmpf"
        assert opts["cluster"] == "spmd"
        assert isinstance(opts["spmd_context"], SPMDContext)


def test_reset_collects_after_options_change(comm: Communicator) -> None:
    """The engine still drives a real query after ``_reset``."""
    with SPMDEngine(
        comm=comm, executor_options={"max_rows_per_partition": 10}
    ) as engine:
        engine._reset(executor_options={"max_rows_per_partition": 3})
        result = pl.LazyFrame({"a": [1, 2, 3, 4, 5]}).collect(engine=engine)
        assert sorted(result["a"].to_list()) == [1, 2, 3, 4, 5]


def test_reset_after_shutdown_raises(comm: Communicator) -> None:
    """``shutdown`` is idempotent; ``_reset`` after shutdown raises every time."""
    engine = SPMDEngine(comm=comm)
    engine.shutdown()
    engine.shutdown()  # idempotent
    with pytest.raises(RuntimeError, match="shut-down"):
        engine._reset()
    with pytest.raises(RuntimeError, match="shut-down"):
        engine._reset()  # still raises on a second attempt
    engine.shutdown()  # still safe after a failed _reset


def test_reset_rejects_construction_time_executor_options(
    comm: Communicator,
) -> None:
    """``_reset`` rejects ``executor_options`` keys read at engine construction."""
    with (
        SPMDEngine(comm=comm) as engine,
        pytest.raises(ValueError, match="num_py_executors"),
    ):
        engine._reset(executor_options={"num_py_executors": 4})


def test_reset_rejects_construction_time_engine_options(
    comm: Communicator,
) -> None:
    """``_reset`` rejects ``engine_options`` keys read at engine construction."""
    from cudf_polars.experimental.rapidsmpf.frontend.hardware_binding import (
        HardwareBindingPolicy,
    )

    with SPMDEngine(comm=comm) as engine:
        with pytest.raises(ValueError, match="hardware_binding"):
            engine._reset(
                engine_options={
                    "hardware_binding": HardwareBindingPolicy(enabled=False),
                },
            )
        with pytest.raises(ValueError, match="memory_resource_config"):
            engine._reset(engine_options={"memory_resource_config": None})
