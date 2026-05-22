# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Dask execution mode."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest
from rapidsmpf.bootstrap import is_running_with_rrun

import polars as pl

from cudf_polars.engine.options import StreamingOptions
from cudf_polars.testing.asserts import assert_gpu_result_equal
from cudf_polars.utils.config import DaskContext

distributed = pytest.importorskip("distributed")

from cudf_polars.engine.dask import DaskEngine  # noqa: E402

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(scope="module")
def engine() -> Iterator[DaskEngine]:
    """Create one Dask/GPU cluster shared across the test module."""
    with DaskEngine(
        # Small partition size so tests exercise the multi-partition code path
        # deterministically, regardless of input size.
        executor_options={"max_rows_per_partition": 10},
    ) as engine:
        yield engine


pytestmark = [
    pytest.mark.skipif(
        is_running_with_rrun(),
        reason="DaskEngine must not be created from within an rrun cluster",
    ),
]


# ---------------------------------------------------------------------------
# GPU tests — share a single Dask cluster for the whole module
# ---------------------------------------------------------------------------


def test_from_options() -> None:
    """DaskEngine.from_options with default StreamingOptions creates a valid engine."""
    opts = StreamingOptions(fallback_mode="silent")
    with DaskEngine.from_options(opts) as engine:
        assert engine.nranks >= 1


def test_yields_engine(engine: DaskEngine) -> None:
    """DaskEngine is a GPUEngine with at least one rank."""
    assert isinstance(engine, pl.GPUEngine)
    assert engine.nranks >= 1


def test_executor_options_forwarded(engine: DaskEngine) -> None:
    """Reserved executor_options keys are injected into the engine config."""
    opts = engine.config["executor_options"]
    assert opts["cluster"] == "dask"
    assert isinstance(opts["dask_context"], DaskContext)


def test_gather_cluster_info(engine: DaskEngine) -> None:
    """gather_cluster_info returns one ClusterInfo per rank with expected fields."""
    infos = engine.gather_cluster_info()
    assert len(infos) == engine.nranks
    for info in infos:
        assert isinstance(info.pid, int)
        assert isinstance(info.hostname, str)
    # Each worker runs in its own process.
    assert len({info.pid for info in infos}) == engine.nranks


def test_worker_host_memory_limit(engine: DaskEngine) -> None:
    """Memory limit is respected."""
    scheduler_info = engine._dask_ctx.client.scheduler_info(n_workers=-1)
    worker = next(iter(scheduler_info["workers"].values()))
    assert worker["memory_limit"] == distributed.system.MEMORY_LIMIT


def test_from_options_creates_engine() -> None:
    """DaskEngine.from_options produces a working engine and runs a query."""
    opts = StreamingOptions(max_rows_per_partition=10, fallback_mode="silent")
    with DaskEngine.from_options(opts) as eng:
        assert isinstance(eng, pl.GPUEngine)
        assert eng.nranks >= 1
        lf = pl.LazyFrame({"a": [1, 2, 3]})
        assert_gpu_result_equal(lf, engine=eng, check_row_order=False)


def test_run(engine: DaskEngine) -> None:
    result = engine._run(os.getpid)
    assert len(set(result)) == engine.nranks


@pytest.fixture(scope="module")
def reset_engine() -> Iterator[DaskEngine]:
    """Module-scoped engine for reset tests — independent of ``engine``.

    These tests exercise :meth:`DaskEngine._reset` (which mutates the
    engine in-place). A dedicated fixture keeps those mutations from
    leaking into the other tests.
    """
    with DaskEngine(
        executor_options={"max_rows_per_partition": 10},
    ) as e:
        yield e


def test_reset_keeps_workers_alive(reset_engine: DaskEngine) -> None:
    """``_reset`` must not respawn dask workers."""
    workers_before = sorted(
        reset_engine._dask_ctx.client.scheduler_info(n_workers=-1)["workers"]
    )
    pids_before = sorted(reset_engine._run(os.getpid))

    reset_engine._reset(executor_options={"max_rows_per_partition": 7})

    workers_after = sorted(
        reset_engine._dask_ctx.client.scheduler_info(n_workers=-1)["workers"]
    )
    pids_after = sorted(reset_engine._run(os.getpid))

    # Same worker addresses …
    assert workers_before == workers_after
    # … and the workers are running in the same OS processes.
    assert pids_before == pids_after


def test_reset_updates_executor_options(reset_engine: DaskEngine) -> None:
    """``_reset`` updates the polars-layer config to the new options."""
    reset_engine._reset(executor_options={"max_rows_per_partition": 42})

    opts = reset_engine.config["executor_options"]
    assert opts["max_rows_per_partition"] == 42
    # Reserved keys are still injected by ``_reset``.
    assert opts["cluster"] == "dask"
    assert isinstance(opts["dask_context"], DaskContext)


def test_reset_collects_after_options_change(reset_engine: DaskEngine) -> None:
    """The engine still drives a real query after ``_reset``."""
    reset_engine._reset(executor_options={"max_rows_per_partition": 3})
    assert_gpu_result_equal(
        pl.LazyFrame({"a": [1, 2, 3, 4, 5]}),
        engine=reset_engine,
        check_row_order=False,
    )


def test_reset_after_shutdown_raises() -> None:
    """``shutdown`` is idempotent; ``_reset`` after shutdown raises every time."""
    engine = DaskEngine(executor_options={"max_rows_per_partition": 10})
    engine.shutdown()
    engine.shutdown()  # idempotent
    with pytest.raises(RuntimeError, match="shut-down"):
        engine._reset()
    with pytest.raises(RuntimeError, match="shut-down"):
        engine._reset()  # still raises on a second attempt
    engine.shutdown()  # still safe after a failed _reset


def test_reset_rejects_construction_time_executor_options(
    reset_engine: DaskEngine,
) -> None:
    """``_reset`` rejects ``executor_options`` keys read at worker setup."""
    with pytest.raises(ValueError, match="num_py_executors"):
        reset_engine._reset(executor_options={"num_py_executors": 4})


def test_reset_rejects_construction_time_engine_options(
    reset_engine: DaskEngine,
) -> None:
    """``_reset`` rejects ``engine_options`` keys read at worker setup."""
    from cudf_polars.engine.hardware_binding import (
        HardwareBindingPolicy,
    )

    with pytest.raises(ValueError, match="hardware_binding"):
        reset_engine._reset(
            engine_options={
                "hardware_binding": HardwareBindingPolicy(enabled=False),
            },
        )
    with pytest.raises(ValueError, match="memory_resource_config"):
        reset_engine._reset(engine_options={"memory_resource_config": None})
