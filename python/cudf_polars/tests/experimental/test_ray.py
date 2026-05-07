# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Ray execution mode."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
from rapidsmpf.bootstrap import is_running_with_rrun

import polars as pl

from cudf_polars.utils.config import RayContext

ray = pytest.importorskip("ray")
from cudf_polars.experimental.rapidsmpf.frontend.ray import RayEngine  # noqa: E402

if TYPE_CHECKING:
    from collections.abc import Iterator

NUM_RANKS = 2


@pytest.fixture(scope="module")
def engine() -> Iterator[RayEngine]:
    """Create one Ray cluster + GPU actors shared across the test session."""
    with RayEngine(
        # Use a small partition size so tests exercise the multi-partition
        # code path deterministically, regardless of input size.
        executor_options={"max_rows_per_partition": 10},
        engine_options={"allow_gpu_sharing": True},
        num_ranks=NUM_RANKS,
        ray_init_options={"include_dashboard": False},
    ) as engine:
        yield engine


pytestmark = [
    pytest.mark.skipif(
        is_running_with_rrun(),
        reason="RayEngine must not be created from within an rrun cluster",
    ),
]


# ---------------------------------------------------------------------------
# Context-manager smoke tests (no GPU required)
# ---------------------------------------------------------------------------


def test_reserved_executor_keys() -> None:
    """executor_options rejects reserved keys."""
    for key in ("cluster", "spmd_context", "ray_context"):
        with pytest.raises(TypeError, match="reserved"):
            RayEngine(executor_options={key: "anything"})


def test_reserved_engine_options_keys() -> None:
    """engine_options rejects reserved keys."""
    for key in ("memory_resource", "executor"):
        with pytest.raises(TypeError, match="reserved"):
            RayEngine(engine_options={key: "anything"})


def test_raises_inside_rrun() -> None:
    """RayEngine must not be created from within an rrun cluster."""
    with (
        patch(
            "rapidsmpf.bootstrap.is_running_with_rrun",
            return_value=True,
        ),
        pytest.raises(RuntimeError, match="rrun"),
    ):
        RayEngine()


def test_num_ranks_requires_allow_gpu_sharing() -> None:
    """num_ranks requires engine_options['allow_gpu_sharing']=True."""
    with pytest.raises(ValueError, match="allow_gpu_sharing"):
        RayEngine(num_ranks=NUM_RANKS)
    with pytest.raises(ValueError, match="allow_gpu_sharing"):
        RayEngine(num_ranks=NUM_RANKS, engine_options={"allow_gpu_sharing": False})


def test_num_ranks_must_be_positive() -> None:
    """num_ranks must be at least 1."""
    with pytest.raises(ValueError, match="num_ranks"):
        RayEngine(num_ranks=0, engine_options={"allow_gpu_sharing": True})


# ---------------------------------------------------------------------------
# GPU tests — share a single Ray cluster + actor set for the whole session
# ---------------------------------------------------------------------------


def test_yields_engine(
    engine: RayEngine,
) -> None:
    """RayEngine is a GPUEngine with at least one rank."""
    assert isinstance(engine, pl.GPUEngine)
    assert engine.nranks >= 1


def test_executor_options_forwarded(
    engine: RayEngine,
) -> None:
    """Reserved executor_options keys are injected into the engine config."""
    opts = engine.config["executor_options"]
    assert opts["cluster"] == "ray"
    assert isinstance(opts["ray_context"], RayContext)
    assert engine.rank_actors == opts["ray_context"].rank_actors
    assert len(engine.rank_actors) == engine.nranks


def test_gather_cluster_info(engine: RayEngine) -> None:
    """gather_cluster_info returns one ClusterInfo per rank with expected fields."""
    infos = engine.gather_cluster_info()
    assert len(infos) == engine.nranks
    for info in infos:
        assert isinstance(info.hostname, str)
        assert isinstance(info.pid, int)
    # Each actor runs in its own process.
    assert len({info.pid for info in infos}) == engine.nranks


def test_scan(engine: RayEngine) -> None:
    """Input rows are partitioned across actors; total output equals input."""
    lf = pl.LazyFrame({"a": [1, 2, 3]})
    result = lf.collect(engine=engine)
    assert result.shape == (3, 1)
    assert sorted(result["a"].to_list()) == [1, 2, 3]


def test_filter(engine: RayEngine) -> None:
    """Filter is applied correctly across all actors."""
    lf = pl.LazyFrame({"a": [1, 2, 3, 4, 5]})
    result = lf.filter(pl.col("a") > 3).collect(engine=engine)
    assert result.shape == (2, 1)
    assert sorted(result["a"].to_list()) == [4, 5]


def test_group_by(engine: RayEngine) -> None:
    """Group-by produces the correct aggregation across all ranks."""
    # max_rows_per_partition=10 (set on the session fixture) gives each rank
    # exactly 5 partitions, so the multi-partition path is always exercised.
    n, n_keys = engine.nranks * 50, 5
    keys = [str(i % n_keys) for i in range(n)]
    vals = list(range(n))
    lf = pl.LazyFrame({"key": keys, "val": vals})
    result = (
        lf.group_by("key").agg(pl.col("val").sum()).collect(engine=engine).sort("key")
    )
    expected = (
        pl.LazyFrame({"key": keys, "val": vals})
        .group_by("key")
        .agg(pl.col("val").sum())
        .collect()
        .sort("key")
    )
    assert result.shape == expected.shape
    assert result["key"].to_list() == expected["key"].to_list()
    assert result["val"].to_list() == expected["val"].to_list()


def test_join(engine: RayEngine) -> None:
    """Hash join between two tables produces the correct result across all ranks."""
    # max_rows_per_partition=10 (set on the session fixture) gives each rank
    # exactly 5 partitions, so the multi-partition path is always exercised.
    n = engine.nranks * 50
    lf_left = pl.LazyFrame({"key": list(range(n)), "val_left": list(range(n))})
    lf_right = pl.LazyFrame(
        {"key": list(range(n)), "val_right": [x * 2 for x in range(n)]}
    )
    result = lf_left.join(lf_right, on="key").collect(engine=engine).sort("key")
    assert result.shape == (n, 3)
    assert result["val_left"].to_list() == list(range(n))
    assert result["val_right"].to_list() == [x * 2 for x in range(n)]


def test_empty_dataframe(engine: RayEngine) -> None:
    """An empty LazyFrame produces an empty result with the correct schema."""
    lf = pl.LazyFrame(
        {"a": pl.Series([], dtype=pl.Int32), "b": pl.Series([], dtype=pl.Float64)}
    )
    result = lf.collect(engine=engine)
    assert result.shape == (0, 2)
    assert result.columns == ["a", "b"]
    assert result.dtypes == [pl.Int32, pl.Float64]


def test_run(engine: RayEngine) -> None:
    result = engine._run(os.getpid)
    assert len(set(result)) == engine.nranks


def test_num_ranks_oversubscribes() -> None:
    """num_ranks creates the requested number of actors sharing GPU 0."""
    n = 2
    with RayEngine(
        executor_options={"max_rows_per_partition": 10},
        engine_options={"allow_gpu_sharing": True},
        num_ranks=n,
        ray_init_options={"include_dashboard": False},
    ) as engine:
        assert engine.nranks == n
        assert len(engine.rank_actors) == n
        result = pl.LazyFrame({"a": [1, 2, 3, 4]}).collect(engine=engine)
        assert sorted(result["a"].to_list()) == [1, 2, 3, 4]


@pytest.fixture(scope="module")
def reset_engine() -> Iterator[RayEngine]:
    """Module-scoped engine for reset tests — independent of ``engine``.

    These tests exercise :meth:`RayEngine._reset` (which mutates the
    engine in-place) and the shutdown guard. A dedicated fixture keeps
    those mutations from leaking into the other tests.
    """
    with RayEngine(
        executor_options={"max_rows_per_partition": 10},
        engine_options={"allow_gpu_sharing": True},
        num_ranks=NUM_RANKS,
        ray_init_options={"include_dashboard": False},
    ) as e:
        yield e


def test_reset_keeps_actors_alive(reset_engine: RayEngine) -> None:
    """``_reset`` must not respawn rank actor processes."""
    actors_before = list(reset_engine.rank_actors)
    pids_before = reset_engine._run(os.getpid)

    reset_engine._reset(
        executor_options={"max_rows_per_partition": 7},
        engine_options={"allow_gpu_sharing": True},
    )

    actors_after = list(reset_engine.rank_actors)
    pids_after = reset_engine._run(os.getpid)

    # The Python ``ActorHandle`` objects are the same instances …
    assert all(a is b for a, b in zip(actors_before, actors_after, strict=True))
    # … and the actors are running in the same OS processes.
    assert pids_before == pids_after


def test_reset_updates_executor_options(reset_engine: RayEngine) -> None:
    """``_reset`` updates the polars-layer config to the new options."""
    reset_engine._reset(
        executor_options={"max_rows_per_partition": 42},
        engine_options={"allow_gpu_sharing": True},
    )

    opts = reset_engine.config["executor_options"]
    assert opts["max_rows_per_partition"] == 42
    # Reserved keys are still injected by ``_reset``.
    assert opts["cluster"] == "ray"
    assert isinstance(opts["ray_context"], RayContext)
    assert opts["ray_context"].rank_actors == reset_engine.rank_actors


def test_reset_collects_after_options_change(reset_engine: RayEngine) -> None:
    """The engine still drives a real query after ``_reset``."""
    reset_engine._reset(
        executor_options={"max_rows_per_partition": 3},
        engine_options={"allow_gpu_sharing": True},
    )
    result = pl.LazyFrame({"a": [1, 2, 3, 4, 5]}).collect(engine=reset_engine)
    assert sorted(result["a"].to_list()) == [1, 2, 3, 4, 5]


def test_reset_after_shutdown_raises() -> None:
    """``shutdown`` is idempotent; ``_reset`` after shutdown raises every time."""
    engine = RayEngine(
        executor_options={"max_rows_per_partition": 10},
        engine_options={"allow_gpu_sharing": True},
        num_ranks=NUM_RANKS,
        ray_init_options={"include_dashboard": False},
    )
    engine.shutdown()
    engine.shutdown()  # idempotent
    with pytest.raises(RuntimeError, match="shut-down"):
        engine._reset()
    with pytest.raises(RuntimeError, match="shut-down"):
        engine._reset()  # still raises on a second attempt
    engine.shutdown()  # still safe after a failed _reset


def test_reset_rejects_construction_time_executor_options(
    reset_engine: RayEngine,
) -> None:
    """``_reset`` rejects ``executor_options`` keys read at actor construction."""
    with pytest.raises(ValueError, match="num_py_executors"):
        reset_engine._reset(
            executor_options={"num_py_executors": 4},
            engine_options={"allow_gpu_sharing": True},
        )


def test_reset_rejects_construction_time_engine_options(
    reset_engine: RayEngine,
) -> None:
    """``_reset`` rejects ``engine_options`` keys read at actor construction."""
    from cudf_polars.experimental.rapidsmpf.frontend.hardware_binding import (
        HardwareBindingPolicy,
    )

    with pytest.raises(ValueError, match="hardware_binding"):
        reset_engine._reset(
            engine_options={
                "allow_gpu_sharing": True,
                "hardware_binding": HardwareBindingPolicy(enabled=False),
            },
        )
    with pytest.raises(ValueError, match="memory_resource_config"):
        reset_engine._reset(
            engine_options={
                "allow_gpu_sharing": True,
                "memory_resource_config": None,
            },
        )


def test_shutdown_skips_when_ray_not_initialized() -> None:
    """``shutdown`` short-circuits if ``ray.is_initialized()`` is ``False``."""
    engine = RayEngine(
        executor_options={"max_rows_per_partition": 10},
        engine_options={"allow_gpu_sharing": True},
        num_ranks=NUM_RANKS,
        ray_init_options={"include_dashboard": False},
    )
    try:
        with patch("ray.is_initialized", return_value=False):
            engine.shutdown()  # must not raise, must not auto-init
        # The exit stack is closed; rank_actors is cleared.
        assert engine._rank_actors is None
    finally:
        # Defensive: if the guard somehow didn't fire, make sure the
        # actors are released so the next test's fixture isn't blocked.
        if engine._rank_actors is not None:
            engine.shutdown()
