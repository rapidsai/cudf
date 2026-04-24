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


@pytest.fixture(scope="module")
def engine() -> Iterator[RayEngine]:
    """Create one Ray cluster + GPU actors shared across the test session."""
    with RayEngine(
        # Use a small partition size so tests exercise the multi-partition
        # code path deterministically, regardless of input size.
        executor_options={"max_rows_per_partition": 10},
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
    for key in ("runtime", "cluster", "spmd_context", "ray_context"):
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
    assert opts["runtime"] == "rapidsmpf"
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
