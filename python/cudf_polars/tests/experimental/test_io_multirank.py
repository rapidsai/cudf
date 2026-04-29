# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""IO tests for the streaming engines."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from rapidsmpf.bootstrap import is_running_with_rrun

import polars as pl

from cudf_polars.experimental.rapidsmpf.frontend.spmd import SPMDEngine
from cudf_polars.testing.asserts import assert_sink_result_equal
from cudf_polars.utils.config import Cluster, StreamingExecutor

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from rapidsmpf.communicator.communicator import Communicator

    from cudf_polars.experimental.rapidsmpf.frontend.core import StreamingEngine

# Runs the spmd variant even under rrun with nranks > 1. The ray/dask
# variants skip themselves in that environment.
pytestmark = [
    pytest.mark.spmd,
]


@pytest.fixture(scope="module")
def df() -> pl.LazyFrame:
    return pl.LazyFrame(
        {
            "x": range(30_000),
            "y": [1, 2, None] * 10_000,
            "z": ["ẅ", "a", "z", "123", "abcd"] * 6_000,
        }
    )


@pytest.fixture(params=["spmd", "ray", "dask"])
def engine(
    request: pytest.FixtureRequest,
    spmd_comm: Communicator,
) -> Iterator[StreamingEngine]:
    """Yield each supported streaming engine."""
    backend = request.param
    executor_options = {"max_rows_per_partition": 1_000}

    if backend == "spmd":
        with SPMDEngine(
            comm=spmd_comm,
            executor_options=executor_options,
        ) as eng:
            yield eng
        return

    if is_running_with_rrun():
        pytest.skip(f"{backend}Engine must not be created from within an rrun cluster")

    if backend == "ray":
        pytest.importorskip("ray", reason="ray is not installed")
        from cudf_polars.experimental.rapidsmpf.frontend.ray import RayEngine

        with RayEngine(
            executor_options=executor_options,
            ray_init_options={"include_dashboard": False},
        ) as eng:
            yield eng
        return

    assert backend == "dask"
    pytest.importorskip("distributed", reason="distributed is not installed")
    from cudf_polars.experimental.rapidsmpf.frontend.dask import DaskEngine

    with DaskEngine(executor_options=executor_options) as eng:
        yield eng


def test_sink_parquet_directory(
    engine: StreamingEngine, df: pl.LazyFrame, tmp_path: Path
) -> None:
    """Streaming engines always sink to a directory of partition files."""
    path = tmp_path / "out.parquet"
    assert_sink_result_equal(df, path, engine=engine)

    gpu_path = path.with_name(f"{path.stem}_gpu{path.suffix}")
    assert gpu_path.is_dir()
    assert any(gpu_path.iterdir())


def test_sink_parquet_empty_rank(engine: StreamingEngine, tmp_path: Path) -> None:
    """A rank that receives no partitions."""
    lazydf = pl.LazyFrame({"x": [1]})
    path = tmp_path / "tiny.parquet"
    assert_sink_result_equal(lazydf, path, engine=engine)

    gpu_path = path.with_name(f"{path.stem}_gpu{path.suffix}")
    assert gpu_path.is_dir()


@pytest.mark.parametrize(
    "cluster",
    [Cluster.SPMD, Cluster.RAY, Cluster.DASK, Cluster.DISTRIBUTED],
)
def test_sink_to_directory_false_raises(cluster: Cluster) -> None:
    """Explicit ``sink_to_directory=False`` is rejected for every multi-rank cluster."""
    with pytest.raises(
        ValueError,
        match=rf"The {cluster.value} cluster requires sink_to_directory=True",
    ):
        StreamingExecutor(cluster=cluster, sink_to_directory=False)
