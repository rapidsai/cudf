# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Tests for SPMD execution mode."""

from __future__ import annotations

from typing import Any

import pytest
from rapidsmpf.bootstrap import is_running_with_rrun

import polars as pl

import rmm.mr

from cudf_polars.experimental.rapidsmpf.collectives.common import reserve_op_id
from cudf_polars.experimental.rapidsmpf.spmd import (
    allgather_polars_dataframe,
    spmd_execution,
)

pytestmark = pytest.mark.skipif(
    not is_running_with_rrun(),
    reason="use `rrun -n <nproc> pytest ...` to run SPMD tests",
)


def test_spmd_execution_yields_context_and_engine() -> None:
    """spmd_execution yields a (Context, GPUEngine) pair."""
    with spmd_execution() as (ctx, engine):
        assert ctx is not None
        assert isinstance(engine, pl.GPUEngine)


def test_spmd_execution_reserved_keys() -> None:
    """executor_options rejects reserved keys."""
    for key in ("runtime", "cluster", "spmd"):
        with (
            pytest.raises(ValueError, match="reserved"),
            spmd_execution(executor_options={key: "anything"}),
        ):
            pass


def test_spmd_execution_engine_kwargs_reserved_keys() -> None:
    """engine_kwargs rejects keys that are set explicitly by spmd_execution."""
    for key in ("raise_on_fail", "memory_resource", "executor"):
        kwargs: dict[str, Any] = {key: "anything"}
        with (
            pytest.raises(ValueError, match="reserved"),
            spmd_execution(**kwargs),
        ):
            pass


def test_spmd_execution_engine_kwargs_parquet_options() -> None:
    """engine_kwargs forwards parquet_options to GPUEngine without error."""
    with spmd_execution(parquet_options={}) as (ctx, engine):
        assert isinstance(engine, pl.GPUEngine)


def test_spmd_execution_custom_mr() -> None:
    """spmd_execution accepts a custom memory resource."""
    mr = rmm.mr.CudaMemoryResource()
    with spmd_execution(mr=mr) as (ctx, engine):
        result = pl.LazyFrame({"a": [1, 2, 3]}).collect(engine=engine)
    assert result.shape == (3, 1)


def test_spmd_execution_scan() -> None:
    """Each rank scans its own single-row LazyFrame and gets that row back."""
    with spmd_execution() as (ctx, engine):
        rank = ctx.comm().rank
        lf = pl.LazyFrame({"a": [rank], "b": [rank * 10]})
        result = lf.collect(engine=engine)
    assert result.shape == (1, 2)
    assert result["a"].to_list() == [rank]
    assert result["b"].to_list() == [rank * 10]


def test_spmd_collect_then_lazy_equivalent() -> None:
    """collect().lazy() preserves SPMD semantics: an intermediate materialize is a no-op.

    In SPMD mode a DataFrame is always rank-local.  When it is wrapped back
    into a LazyFrame the engine processes that rank's copy in full rather than
    re-slicing it across ranks.  So ``lf.collect().lazy().op.collect()`` must
    produce the same result as ``lf.op.collect()``.
    """
    with spmd_execution() as (ctx, engine):
        rank = ctx.comm().rank
        lf = pl.LazyFrame({"a": [rank, rank + 1, rank + 2], "b": [0, 1, 2]})

        # One-step
        one_step = lf.filter(pl.col("b") >= 1).collect(engine=engine)

        # Two-step: materialize then re-wrap
        intermediate = lf.collect(engine=engine)
        two_step = intermediate.lazy().filter(pl.col("b") >= 1).collect(engine=engine)

    assert one_step.sort("a").equals(two_step.sort("a"))


def test_spmd_execution_group_by() -> None:
    """Group-by on rank-local data, then allgather to verify the global result."""
    with spmd_execution() as (ctx, engine):
        rank = ctx.comm().rank
        nranks = ctx.comm().nranks
        lf = pl.LazyFrame({"a": [rank], "b": [rank * 10]})
        local_result = lf.group_by("a").agg(pl.col("b").sum()).collect(engine=engine)
        with reserve_op_id() as op_id:
            global_result = allgather_polars_dataframe(
                ctx=ctx, local_df=local_result, op_id=op_id
            )
    assert global_result.shape == (nranks, 2)
    assert global_result.sort("a")["a"].to_list() == list(range(nranks))
    assert global_result.sort("a")["b"].to_list() == [r * 10 for r in range(nranks)]


def test_allgather_polars_dataframe() -> None:
    """allgather_polars_dataframe collects every rank's contribution in rank order."""
    with spmd_execution() as (ctx, _):
        rank = ctx.comm().rank
        nranks = ctx.comm().nranks
        local = pl.DataFrame({"rank": [rank], "val": [rank * 2]})
        with reserve_op_id() as op_id:
            result = allgather_polars_dataframe(ctx=ctx, local_df=local, op_id=op_id)
    assert result.shape == (nranks, 2)
    assert result["rank"].to_list() == list(range(nranks))
    assert result["val"].to_list() == [r * 2 for r in range(nranks)]


def test_spmd_execution_max_workers() -> None:
    """executor_options forwards rapidsmpf_py_executor_max_workers to the thread pool."""
    with spmd_execution(executor_options={"rapidsmpf_py_executor_max_workers": 2}) as (
        ctx,
        engine,
    ):
        result = pl.LazyFrame({"a": [1, 2, 3]}).collect(engine=engine)
    assert result.shape == (3, 1)


def test_allgather_polars_dataframe_multi_column() -> None:
    """allgather preserves column names, count, and dtypes for multi-column DataFrames."""
    with spmd_execution() as (ctx, _):
        rank = ctx.comm().rank
        nranks = ctx.comm().nranks
        local = pl.DataFrame(
            {"rank": [rank], "x": [float(rank)], "label": [f"r{rank}"]}
        )
        with reserve_op_id() as op_id:
            result = allgather_polars_dataframe(ctx=ctx, local_df=local, op_id=op_id)
    assert result.shape == (nranks, 3)
    assert result.columns == ["rank", "x", "label"]
    sorted_result = result.sort("rank")
    assert sorted_result["rank"].to_list() == list(range(nranks))
    assert sorted_result["x"].to_list() == [float(r) for r in range(nranks)]
    assert sorted_result["label"].to_list() == [f"r{r}" for r in range(nranks)]
