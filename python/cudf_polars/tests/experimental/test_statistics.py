# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ``gather_statistics`` / ``global_statistics`` on streaming engines."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from rapidsmpf.bootstrap import is_running_with_rrun
from rapidsmpf.config import Options
from rapidsmpf.statistics import Statistics

import polars as pl

from cudf_polars.experimental.rapidsmpf.frontend.spmd import SPMDEngine

if TYPE_CHECKING:
    from collections.abc import Iterator

    from rapidsmpf.communicator.communicator import Communicator

    from cudf_polars.experimental.rapidsmpf.frontend.core import StreamingEngine

# Runs the spmd variant even under rrun with nranks > 1. The ray/dask
# variants skip themselves in that environment.
pytestmark = [
    pytest.mark.spmd,
    # Ray's subprocess management and distributed's shutdown leak unclosed
    # /dev/null handles and sockets; suppress the noise.
    pytest.mark.filterwarnings("ignore::ResourceWarning"),
]


@pytest.fixture(params=["spmd", "ray", "dask"])
def engine(
    request: pytest.FixtureRequest,
    spmd_comm: Communicator,
) -> Iterator[StreamingEngine]:
    """Yield each supported streaming engine with statistics enabled."""
    backend = request.param
    rapidsmpf_options = Options({"statistics": "True"})
    executor_options = {"max_rows_per_partition": 10}

    if backend == "spmd":
        with SPMDEngine(
            comm=spmd_comm,
            rapidsmpf_options=rapidsmpf_options,
            executor_options=executor_options,
        ) as engine:
            yield engine
        return

    if is_running_with_rrun():
        pytest.skip(f"{backend}Engine must not be created from within an rrun cluster")

    if backend == "ray":
        pytest.importorskip("ray", reason="ray is not installed")
        from cudf_polars.experimental.rapidsmpf.frontend.ray import RayEngine

        with RayEngine(
            rapidsmpf_options=rapidsmpf_options,
            executor_options=executor_options,
            ray_init_options={"include_dashboard": False},
        ) as engine:
            yield engine
        return

    assert backend == "dask"
    pytest.importorskip("distributed", reason="distributed is not installed")
    from cudf_polars.experimental.rapidsmpf.frontend.dask import DaskEngine

    with DaskEngine(
        rapidsmpf_options=rapidsmpf_options,
        executor_options=executor_options,
    ) as engine:
        yield engine


def test_gather_and_clear_statistics(engine: StreamingEngine) -> None:
    """gather_statistics returns one enabled Statistics per rank; clear=True empties."""
    stats = engine.gather_statistics()
    assert len(stats) == engine.nranks
    for s in stats:
        assert isinstance(s, Statistics)
        assert s.enabled

    # Drive a group_by so the streaming pipeline executes a real shuffle
    # and the per-rank statistics get populated.
    n, n_keys = engine.nranks * 50, 5
    lf = pl.LazyFrame(
        {"key": [str(i % n_keys) for i in range(n)], "val": list(range(n))}
    )
    lf.group_by("key").agg(pl.col("val").sum()).collect(engine=engine)

    # Gather with clear=True captures the current stats and then empties them.
    engine.gather_statistics(clear=True)

    stats = engine.gather_statistics()
    assert len(stats) == engine.nranks
    for s in stats:
        assert s.enabled
        assert s.list_stat_names() == []


def test_global_statistics(engine: StreamingEngine) -> None:
    """global_statistics returns a single Statistics merged across all ranks."""
    n, n_keys = engine.nranks * 50, 5
    lf = pl.LazyFrame(
        {"key": [str(i % n_keys) for i in range(n)], "val": list(range(n))}
    )
    lf.group_by("key").agg(pl.col("val").sum()).collect(engine=engine)

    # Warm up: gather_statistics itself triggers host allocations that get
    # recorded into ctx.statistics(). After the first call, stat names are
    # stable across subsequent gathers.
    engine.gather_statistics()

    merged = engine.global_statistics()
    assert isinstance(merged, Statistics)
    assert merged.enabled
    assert merged.list_stat_names() != []

    # Each rank's stat names must be a subset of the merged names.
    per_rank = engine.gather_statistics()
    for s in per_rank:
        assert set(s.list_stat_names()) <= set(merged.list_stat_names())
