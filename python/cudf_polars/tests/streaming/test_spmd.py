# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for SPMD execution mode."""

from __future__ import annotations

import os
import uuid
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

import polars as pl

import rmm.mr
from rapidsmpf.bootstrap import is_running_with_rrun
from rapidsmpf.rmm_resource_adaptor import RmmResourceAdaptor

import cudf_polars.quent
import cudf_polars.quent._context
from cudf_polars.engine.core import _find_memory_error
from cudf_polars.engine.hardware_binding import HardwareBindingPolicy
from cudf_polars.engine.options import StreamingOptions
from cudf_polars.engine.spmd import (
    SPMDEngine,
    allgather_polars_dataframe,
)
from cudf_polars.streaming.actor_graph.collectives.common import reserve_op_id
from cudf_polars.testing.asserts import assert_gpu_result_equal
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
    for key in ("cluster", "spmd_context"):
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


def test_sort_slice_over_union_of_duplicated_streams(
    spmd_engine: SPMDEngine,
) -> None:
    """Sort+head over a concat of two group-by branches returns the global result on every rank."""
    lf1 = (
        pl.LazyFrame({"name": ["alice"], "score": [1.0]})
        .group_by("name")
        .agg(pl.col("score").sum())
    )
    lf2 = (
        pl.LazyFrame({"name": ["bob"], "score": [2.0]})
        .group_by("name")
        .agg(pl.col("score").sum())
    )
    lf = pl.concat([lf1, lf2]).sort("score").head(10)
    assert_gpu_result_equal(lf, engine=spmd_engine, check_row_order=False)


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
    with SPMDEngine(comm=comm) as engine:
        with pytest.raises(ValueError, match="hardware_binding"):
            engine._reset(
                engine_options={
                    "hardware_binding": HardwareBindingPolicy(enabled=False),
                },
            )
        with pytest.raises(ValueError, match="memory_resource_config"):
            engine._reset(engine_options={"memory_resource_config": None})


def test_quent_context_user_provided(spmd_engine: SPMDEngine) -> None:
    # Ensure that the user-provided quent context is used if provided
    quent_context = cudf_polars.quent._context.QuentContext(
        engine=cudf_polars.quent.Engine(
            id=uuid.uuid4(),
            implementation=cudf_polars.quent.Implementation(
                name="test_implementation", version="0.0.0"
            ),
        ),
        query_group=cudf_polars.quent.QueryGroup(instance_name="test_query_group"),
        query=cudf_polars.quent.Query(instance_name="test_query"),
    )

    with SPMDEngine(
        comm=spmd_engine.comm, executor_options={"quent_context": quent_context}
    ) as engine:
        assert engine.config["executor_options"]["quent_context"] == quent_context


def test_quent_context_default(spmd_engine: SPMDEngine) -> None:
    with SPMDEngine(comm=spmd_engine.comm) as engine:
        assert engine.config["executor_options"].get("quent_context") is None


# Group keys probed with num_partitions=2, nranks=2, ROUND_ROBIN:
#   _SAME_RANK_KEYS[r] hashes to partition r: data stays on its origin rank.
#   _CROSS_RANK_KEYS[r] hashes to partition 1-r: data is fully shuffled away.
# num_partitions=2 = max(nranks=2, local_count=1).  local_count=1 requires
# max_rows_per_partition >= the number of rows per rank (3 here), so we use 4.
_SAME_RANK_KEYS = [
    0,
    3,
]  # g=0 hashes to partition 0 (rank 0); g=3 hashes to partition 1 (rank 1)
_CROSS_RANK_KEYS = [
    3,
    0,
]  # g=3 hashes to partition 1 (rank 1); g=0 hashes to partition 0 (rank 0)


@pytest.mark.parametrize(
    "expr,is_scalar",
    [
        (pl.col("x").sum().over("g").alias("result"), True),
        (pl.col("x").rank(method="dense").over("g").alias("result"), False),
    ],
    ids=["scalar_sum", "nonscalar_rank"],
)
@pytest.mark.parametrize(
    "cross_rank",
    [False, True],
    ids=["same_rank", "cross_rank"],
)
def test_over_multirank(
    request: pytest.FixtureRequest,
    comm: Communicator,
    expr: pl.Expr,
    is_scalar: bool,  # noqa: FBT001
    cross_rank: bool,  # noqa: FBT001
) -> None:
    """over() correctness in multi-rank SPMD mode, same-rank and cross-rank cases.

    same_rank: group keys hash to the origin rank's own partition (happy path).
    cross_rank: group keys hash to the other rank's partition, exercising the
    bug where row_idx spaces are rank-local so Phase 2 fills the wrong
    accumulated slots and each rank receives the other rank's data.

    max_rows_per_partition=4 keeps all 3 rows in one chunk (local_count=1),
    so num_partitions=max(nranks=2, 1)=2, matching the probed key assignments.
    """
    with SPMDEngine(
        comm=comm,
        executor_options={"max_rows_per_partition": 4, "dynamic_planning": {}},
    ) as engine:
        rank = engine.rank
        nranks = engine.nranks
        if nranks != 2:
            request.applymarker(
                pytest.mark.skip(
                    reason="key assignments are probed for exactly 2 ranks"
                )
            )
        keys = _CROSS_RANK_KEYS if cross_rank else _SAME_RANK_KEYS
        g = keys[rank]
        xs = [rank * 3 + 1, rank * 3 + 2, rank * 3 + 3]
        lf = pl.LazyFrame({"g": [g, g, g], "x": xs})
        local_result = lf.select(pl.col("g"), pl.col("x"), expr).collect(engine=engine)

        # Each rank must get back its OWN rows (not another rank's).
        assert local_result["g"].unique().to_list() == [g], (
            f"rank {rank}: expected only group {g} in output, "
            f"got {local_result['g'].unique().to_list()}"
        )

        with reserve_op_id() as op_id:
            global_result = allgather_polars_dataframe(
                engine=engine, local_df=local_result, op_id=op_id
            )

        assert global_result.shape == (3 * nranks, 3)
        for r in range(nranks):
            grp_g = keys[r]
            grp = global_result.filter(pl.col("g") == grp_g).sort("x")
            assert grp.shape == (3, 3), f"rank {r} group has wrong row count"
            expected_xs = [r * 3 + 1, r * 3 + 2, r * 3 + 3]
            assert grp["x"].to_list() == expected_xs
            if is_scalar:
                assert grp["result"].to_list() == [sum(expected_xs)] * 3
            else:
                assert grp["result"].to_list() == [1, 2, 3]


def test_over_nonscalar_duplicated_input(
    request: pytest.FixtureRequest,
    comm: Communicator,
) -> None:
    """Non-scalar over() on duplicated=True input produces correct row count and values.

    group_by() AllGathers its result onto all ranks (duplicated=True).  The
    non-scalar over() path must output duplicated=False and only insert rows on
    rank 0, otherwise all ranks insert the same rows (N-fold overcounting) and
    the downstream Repartition skips AllGather.

    max_rows_per_partition=10 keeps all 3 rows in one chunk (local_count=1),
    so modulus=max(nranks=2, 1)=2, matching the _SAME_RANK_KEYS assignments.
    """
    with SPMDEngine(
        comm=comm,
        executor_options={"max_rows_per_partition": 10, "dynamic_planning": {}},
    ) as engine:
        rank = engine.rank
        nranks = engine.nranks
        if nranks != 2:
            request.applymarker(
                pytest.mark.skip(
                    reason="key assignments are probed for exactly 2 ranks"
                )
            )

        coarse_g = _SAME_RANK_KEYS[rank]
        fine_gs = [rank * 3 + 1, rank * 3 + 2, rank * 3 + 3]
        xs = [rank * 30 + 10, rank * 30 + 20, rank * 30 + 30]
        lf = pl.LazyFrame({"fine_g": fine_gs, "coarse_g": [coarse_g] * 3, "x": xs})
        local_result = (
            lf.group_by("fine_g", "coarse_g")
            .agg(pl.col("x").first())
            .with_columns(
                pl.col("x").rank(method="dense").over("coarse_g").alias("rank_x")
            )
            .collect(engine=engine)
        )

        with reserve_op_id() as op_id:
            global_result = allgather_polars_dataframe(
                engine=engine, local_df=local_result, op_id=op_id
            )

        assert global_result.shape == (3 * nranks, 4)
        for r in range(nranks):
            cg = _SAME_RANK_KEYS[r]
            grp = global_result.filter(pl.col("coarse_g") == cg).sort("x")
            assert grp.shape == (3, 4), f"coarse_g={cg}: wrong row count"
            assert grp["rank_x"].to_list() == [1, 2, 3], (
                f"coarse_g={cg}: expected dense ranks [1, 2, 3] "
                f"but got {grp['rank_x'].to_list()}"
            )


def test_find_memory_error() -> None:
    err = MemoryError("oom")
    assert _find_memory_error(err) is err

    inner = MemoryError("oom")
    assert _find_memory_error(BaseExceptionGroup("g", [inner])) is inner

    inner = MemoryError("oom")
    assert (
        _find_memory_error(
            BaseExceptionGroup(
                "outer", [BaseExceptionGroup("inner", [ValueError("x"), inner])]
            )
        )
        is inner
    )

    assert _find_memory_error(BaseExceptionGroup("g", [ValueError("x")])) is None


@pytest.mark.spmd
def test_memory_error_hint(spmd_engine: SPMDEngine) -> None:
    """MemoryError from the actor network is re-raised with a configuration hint."""
    q = pl.LazyFrame({"a": [1, 2, 3]}).select(pl.col("a") + 1)

    for exc in [
        MemoryError("CUDA out of memory"),
        BaseExceptionGroup("unhandled errors", [MemoryError("CUDA out of memory")]),
    ]:
        with (
            patch(
                "cudf_polars.engine.core.run_actor_network",
                side_effect=exc,
            ),
            pytest.raises(MemoryError, match="target_partition_size"),
        ):
            q.collect(engine=spmd_engine)
