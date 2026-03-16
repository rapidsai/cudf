# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Ray execution mode."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import pytest

import polars as pl

from cudf_polars.utils.config import RayContext

ray = pytest.importorskip("ray")

from cudf_polars.experimental.rapidsmpf.frontend.ray import (  # noqa: E402
    RayClient,
    ray_execution,
)

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(scope="session")
def _ray_env() -> Iterator[tuple[RayClient, pl.GPUEngine]]:
    """Create one Ray cluster + GPU actors shared across the test session."""
    try:
        with ray_execution(
            # Use a small partition size so tests exercise the multi-partition
            # code path deterministically, regardless of input size.
            executor_options={"max_rows_per_partition": 10},
            ray_init_options={"include_dashboard": False},
        ) as (
            ray_client,
            engine,
        ):
            yield ray_client, engine
    except RuntimeError as e:
        pytest.skip(f"Ray GPU cluster unavailable: {e}")


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
    for key in ("runtime", "cluster", "spmd", "ray_context"):
        with pytest.raises(TypeError, match="reserved"):
            ray_execution(executor_options={key: "anything"})


def test_ray_execution_reserved_engine_options_keys() -> None:
    """engine_options rejects keys that are set explicitly by ray_execution."""
    for key in ("memory_resource", "executor"):
        kwargs: dict[str, Any] = {key: "anything"}
        with pytest.raises(TypeError, match="reserved"):
            ray_execution(engine_options=kwargs)


def test_ray_client_shutdown_idempotent() -> None:
    """RayClient.shutdown() is safe to call more than once."""
    mock_engine = MagicMock(spec=pl.GPUEngine)
    mock_engine.config = {"executor_options": {"ray_context": RayContext([])}}
    client = RayClient(mock_engine, owns_ray=False)
    client.shutdown()
    client.shutdown()  # must not raise


def test_ray_client_post_shutdown_state() -> None:
    """After shutdown, rank_actors, nranks, and engine all raise RuntimeError."""
    mock_engine = MagicMock(spec=pl.GPUEngine)
    mock_engine.config = {"executor_options": {"ray_context": RayContext([])}}
    client = RayClient(mock_engine, owns_ray=False)
    client.shutdown()
    with pytest.raises(RuntimeError, match="shutdown"):
        _ = client.rank_actors
    with pytest.raises(RuntimeError, match="shutdown"):
        _ = client.nranks
    with pytest.raises(RuntimeError, match="shutdown"):
        _ = client.engine


def test_ray_execution_raises_inside_rrun() -> None:
    """ray_execution() must not be called from within an rrun cluster."""
    with (
        patch(
            "rapidsmpf.bootstrap.is_running_with_rrun",
            return_value=True,
        ),
        pytest.raises(RuntimeError, match="rrun"),
    ):
        ray_execution()


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


def test_ray_execution_executor_options_forwarded(
    ray_client: RayClient,
    engine: pl.GPUEngine,
) -> None:
    """Reserved executor_options keys are injected into the engine config."""
    opts = engine.config["executor_options"]
    assert opts["runtime"] == "rapidsmpf"
    assert opts["cluster"] == "ray"
    assert isinstance(opts["ray_context"], RayContext)
    assert ray_client.rank_actors == opts["ray_context"].rank_actors
    assert len(ray_client.rank_actors) == ray_client.nranks


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


def test_ray_execution_group_by(ray_client: RayClient, engine: pl.GPUEngine) -> None:
    """Group-by produces the correct aggregation across all ranks."""
    # max_rows_per_partition=10 (set on the session fixture) gives each rank
    # exactly 5 partitions, so the multi-partition path is always exercised.
    n, n_keys = ray_client.nranks * 50, 5
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


def test_ray_execution_join(ray_client: RayClient, engine: pl.GPUEngine) -> None:
    """Hash join between two tables produces the correct result across all ranks."""
    # max_rows_per_partition=10 (set on the session fixture) gives each rank
    # exactly 5 partitions, so the multi-partition path is always exercised.
    n = ray_client.nranks * 50
    lf_left = pl.LazyFrame({"key": list(range(n)), "val_left": list(range(n))})
    lf_right = pl.LazyFrame(
        {"key": list(range(n)), "val_right": [x * 2 for x in range(n)]}
    )
    result = lf_left.join(lf_right, on="key").collect(engine=engine).sort("key")
    assert result.shape == (n, 3)
    assert result["val_left"].to_list() == list(range(n))
    assert result["val_right"].to_list() == [x * 2 for x in range(n)]


def test_ray_execution_empty_dataframe(engine: pl.GPUEngine) -> None:
    """An empty LazyFrame produces an empty result with the correct schema."""
    lf = pl.LazyFrame(
        {"a": pl.Series([], dtype=pl.Int32), "b": pl.Series([], dtype=pl.Float64)}
    )
    result = lf.collect(engine=engine)
    assert result.shape == (0, 2)
    assert result.columns == ["a", "b"]
    assert result.dtypes == [pl.Int32, pl.Float64]
