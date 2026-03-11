# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Ray execution mode."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

import polars as pl

ray = pytest.importorskip("ray")

from cudf_polars.experimental.rapidsmpf.ray import (  # noqa: E402
    RayClient,
    ray_execution,
)

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(scope="session")
def _ray_env() -> Iterator[tuple[RayClient, pl.GPUEngine]]:
    """Create one Ray cluster + GPU actors shared across the test session."""
    with ray_execution(ray_init_kwargs={"include_dashboard": False}) as (
        ray_client,
        engine,
    ):
        yield ray_client, engine


@pytest.fixture(scope="session")
def ray_client(_ray_env: tuple[RayClient, pl.GPUEngine]) -> RayClient:
    """Session-scoped Ray cluster client."""
    return _ray_env[0]


@pytest.fixture(scope="session")
def engine(_ray_env: tuple[RayClient, pl.GPUEngine]) -> pl.GPUEngine:
    """Session-scoped GPU engine backed by the Ray cluster."""
    return _ray_env[1]


pytestmark = [
    # Ray's internal subprocess management leaks /dev/null file handles;
    # suppress the resulting ResourceWarning noise from its internals.
    pytest.mark.filterwarnings("ignore::ResourceWarning"),
]


# ---------------------------------------------------------------------------
# Context-manager smoke tests (no GPU required)
# ---------------------------------------------------------------------------


def test_ray_execution_reserved_executor_keys() -> None:
    """executor_options rejects reserved keys."""
    for key in ("runtime", "cluster", "spmd", "ray_client"):
        with (
            pytest.raises(ValueError, match="reserved"),
            ray_execution(executor_options={key: "anything"}),
        ):
            pass


def test_ray_execution_reserved_engine_kwargs_keys() -> None:
    """engine_kwargs rejects keys that are set explicitly by ray_execution."""
    for key in ("memory_resource", "executor"):
        kwargs: dict[str, Any] = {key: "anything"}
        with (
            pytest.raises(ValueError, match="reserved"),
            ray_execution(engine_kwargs=kwargs),
        ):
            pass


# ---------------------------------------------------------------------------
# GPU tests — share a single Ray cluster + actor set for the whole session
# ---------------------------------------------------------------------------


def test_ray_execution_yields_client_and_engine(
    ray_client: RayClient,
    engine: pl.GPUEngine,
) -> None:
    """ray_execution yields a (RayClient, GPUEngine) pair."""
    assert isinstance(ray_client, RayClient)
    assert isinstance(engine, pl.GPUEngine)
    assert ray_client.nranks >= 1


def test_gather_cluster_info(ray_client: RayClient) -> None:
    """gather_cluster_info returns one info dict per rank with expected fields."""
    infos = ray_client.gather_cluster_info()
    assert len(infos) == ray_client.nranks
    for info in infos:
        assert "node_id" in info
        assert "hostname" in info
        assert "pid" in info
        assert "cuda_visible_devices" in info
        assert isinstance(info["pid"], int)
    # Each actor runs in its own process.
    assert len({info["pid"] for info in infos}) == ray_client.nranks


def test_ray_execution_scan(engine: pl.GPUEngine) -> None:
    """Input rows are partitioned across actors; total output equals input."""
    lf = pl.LazyFrame({"a": [1, 2, 3]})
    result = lf.collect(engine=engine)
    assert result.shape == (3, 1)
    assert sorted(result["a"].to_list()) == [1, 2, 3]


def test_ray_execution_filter(engine: pl.GPUEngine) -> None:
    """Filter is applied correctly across all actors."""
    lf = pl.LazyFrame({"a": [1, 2, 3, 4, 5]})
    result = lf.filter(pl.col("a") > 3).collect(engine=engine)
    assert result.shape == (2, 1)
    assert sorted(result["a"].to_list()) == [4, 5]


def test_ray_execution_group_by(engine: pl.GPUEngine) -> None:
    """Group-by produces the correct global aggregation across all actors."""
    lf = pl.LazyFrame({"key": ["a", "a", "b"], "val": [1, 2, 3]})
    result = (
        lf.group_by("key").agg(pl.col("val").sum()).collect(engine=engine).sort("key")
    )
    assert result.shape == (2, 2)
    assert result["key"].to_list() == ["a", "b"]
    assert result["val"].to_list() == [3, 3]


def test_ray_execution_empty_dataframe(engine: pl.GPUEngine) -> None:
    """An empty LazyFrame produces an empty result with the correct schema."""
    lf = pl.LazyFrame(
        {"a": pl.Series([], dtype=pl.Int32), "b": pl.Series([], dtype=pl.Float64)}
    )
    result = lf.collect(engine=engine)
    assert result.shape == (0, 2)
    assert result.columns == ["a", "b"]
    assert result.dtypes == [pl.Int32, pl.Float64]
