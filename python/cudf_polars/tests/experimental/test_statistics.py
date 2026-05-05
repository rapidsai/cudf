# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ``gather_statistics`` / ``global_statistics`` on streaming engines."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from rapidsmpf.bootstrap import is_running_with_rrun
from rapidsmpf.config import Options
from rapidsmpf.statistics import Statistics

from cudf_polars.experimental.rapidsmpf.frontend.spmd import SPMDEngine

if TYPE_CHECKING:
    from collections.abc import Iterator

    from cudf_polars.experimental.rapidsmpf.frontend.core import StreamingEngine

# Runs the spmd variant even under rrun with nranks > 1. The ray/dask
# variants skip themselves in that environment.
pytestmark = [
    pytest.mark.spmd,
]


@pytest.fixture(params=["spmd", "ray", "dask"])
def engine(
    request: pytest.FixtureRequest,
    spmd_engine: SPMDEngine,
) -> Iterator[StreamingEngine]:
    """Yield each supported streaming engine with statistics enabled."""
    backend = request.param
    rapidsmpf_options = Options({"statistics": "True"})
    executor_options = {"max_rows_per_partition": 10}

    if backend == "spmd":
        with SPMDEngine(
            comm=spmd_engine.comm,
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


def test_statistics(engine: StreamingEngine) -> None:
    """gather_statistics / global_statistics / clear round-trip."""
    # gather_statistics returns one enabled Statistics per rank.
    stats = engine.gather_statistics()
    assert len(stats) == engine.nranks
    for s in stats:
        assert isinstance(s, Statistics)
        assert s.enabled

    # global_statistics returns a single merged, enabled Statistics.
    merged = engine.global_statistics()
    assert isinstance(merged, Statistics)
    assert merged.enabled

    # gather_statistics(clear=True) captures and then empties each rank.
    engine.gather_statistics(clear=True)
    stats = engine.gather_statistics()
    assert len(stats) == engine.nranks
    for s in stats:
        assert s.enabled
        assert s.list_stat_names() == []
