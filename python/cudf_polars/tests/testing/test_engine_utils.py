# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

from cudf_polars.testing.engine_utils import (
    EngineFixtureParam,
    create_streaming_options,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from rapidsmpf.communicator.communicator import Communicator


def test_engine_fixture_param_in_memory():
    param = EngineFixtureParam("in-memory")
    assert param.engine_name == "in-memory"
    assert param.blocksize_mode == "medium"


def test_engine_fixture_param_medium_blocksize():
    param = EngineFixtureParam("spmd")
    assert param.engine_name == "spmd"
    assert param.blocksize_mode == "medium"


def test_engine_fixture_param_small_blocksize():
    param = EngineFixtureParam("spmd-small")
    assert param.engine_name == "spmd"
    assert param.blocksize_mode == "small"


def test_create_streaming_options_medium():
    pytest.importorskip("rapidsmpf")
    opts = create_streaming_options("medium")
    assert opts.max_rows_per_partition == 50
    assert opts.target_partition_size == 1_000_000
    assert opts.raise_on_fail is True


def test_create_streaming_options_small():
    pytest.importorskip("rapidsmpf")
    opts = create_streaming_options("small")
    assert opts.max_rows_per_partition == 4
    assert opts.target_partition_size == 10


def test_create_streaming_options_overrides_merge():
    """Overrides take precedence over the blocksize baseline."""
    pytest.importorskip("rapidsmpf")
    from cudf_polars.experimental.rapidsmpf.frontend.options import StreamingOptions

    overrides = StreamingOptions(max_rows_per_partition=999)
    merged = create_streaming_options("medium", overrides)
    # Override wins.
    assert merged.max_rows_per_partition == 999
    # Untouched baseline field is preserved.
    assert merged.target_partition_size == 1_000_000


# ---------------------------------------------------------------------------
# Single-slot RayEngine cache: ``build_streaming_engine`` reuses one
# :class:`RayEngine` across consecutive ray-backend calls via
# :meth:`RayEngine._reset` to amortize actor-fork + UCXX bootstrap.
# ---------------------------------------------------------------------------

ray = pytest.importorskip("ray")
from rapidsmpf.bootstrap import is_running_with_rrun  # noqa: E402

from cudf_polars.experimental.rapidsmpf.frontend.options import (  # noqa: E402
    StreamingOptions,
)
from cudf_polars.experimental.rapidsmpf.frontend.ray import RayEngine  # noqa: E402
from cudf_polars.testing.engine_utils import (  # noqa: E402
    NUM_RANKS,
    build_streaming_engine,
    shutdown_streaming_engine_cache,
)

_ray_cache_skip = pytest.mark.skipif(
    is_running_with_rrun(),
    reason="RayEngine must not be created from within an rrun cluster",
)


@pytest.fixture
def clean_cache() -> Iterator[None]:
    """Ensure a pristine cache before and after each test."""
    shutdown_streaming_engine_cache()
    try:
        yield
    finally:
        shutdown_streaming_engine_cache()


@pytest.fixture(scope="module")
def _ray_session() -> Iterator[None]:
    """One ``ray.init`` for the whole module so individual tests don't pay
    the bootstrap cost more than once.
    """
    import tempfile

    if not ray.is_initialized():
        os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")
        # Per-process temp dir isolates this module's ray cluster from
        # any concurrent pytest invocations sharing ``/tmp/ray``.
        temp_dir = tempfile.mkdtemp(prefix=f"ray-pytest-{os.getpid()}-")
        ray.init(include_dashboard=False, _temp_dir=temp_dir)
    try:
        yield
    finally:
        # Release any cached engine before tearing down the cluster so
        # the engine's actor handles are released against a still-live
        # cluster (the same teardown-order rule as the conftest).
        shutdown_streaming_engine_cache()
        ray.shutdown()


_RAY_PARAM = EngineFixtureParam(full_name="ray")
_SPMD_PARAM = EngineFixtureParam(full_name="spmd")


@_ray_cache_skip
@pytest.mark.usefixtures("_ray_session", "clean_cache")
def test_first_call_builds_fresh_ray_engine() -> None:
    """The first ``build_streaming_engine`` call for ``"ray"`` builds an engine."""
    engine = build_streaming_engine(_RAY_PARAM, spmd_comm=None)
    assert isinstance(engine, RayEngine)
    # Built with the cache's ``NUM_RANKS`` baseline.
    assert engine.nranks == NUM_RANKS


@_ray_cache_skip
@pytest.mark.usefixtures("_ray_session", "clean_cache")
def test_second_call_reuses_cached_engine() -> None:
    """Subsequent ray calls return the same engine and same actor processes."""
    e1 = build_streaming_engine(_RAY_PARAM, spmd_comm=None)
    assert isinstance(e1, RayEngine)
    pids_before = e1._run(os.getpid)
    actors_before = list(e1.rank_actors)

    e2 = build_streaming_engine(
        _RAY_PARAM,
        spmd_comm=None,
        options=StreamingOptions(max_rows_per_partition=42),
    )
    assert isinstance(e2, RayEngine)

    assert e2 is e1
    actors_after = list(e2.rank_actors)
    pids_after = e2._run(os.getpid)
    assert all(a is b for a, b in zip(actors_before, actors_after, strict=True))
    assert pids_before == pids_after


@_ray_cache_skip
@pytest.mark.usefixtures("_ray_session", "clean_cache")
def test_reset_propagates_options() -> None:
    """The polars-layer config reflects the most recent ``options`` arg."""
    build_streaming_engine(
        _RAY_PARAM,
        spmd_comm=None,
        options=StreamingOptions(max_rows_per_partition=10),
    )
    engine = build_streaming_engine(
        _RAY_PARAM,
        spmd_comm=None,
        options=StreamingOptions(max_rows_per_partition=99),
    )
    assert engine.config["executor_options"]["max_rows_per_partition"] == 99


@_ray_cache_skip
@pytest.mark.usefixtures("_ray_session", "clean_cache")
def test_shutdown_cache_clears_slot() -> None:
    """``shutdown_streaming_engine_cache`` releases the slot; next call rebuilds."""
    e1 = build_streaming_engine(_RAY_PARAM, spmd_comm=None)
    assert isinstance(e1, RayEngine)
    pids_before = e1._run(os.getpid)

    shutdown_streaming_engine_cache()

    e2 = build_streaming_engine(_RAY_PARAM, spmd_comm=None)
    assert isinstance(e2, RayEngine)
    assert e2 is not e1
    pids_after = e2._run(os.getpid)
    # New engine = new actor processes.
    assert set(pids_before).isdisjoint(set(pids_after))


@_ray_cache_skip
@pytest.mark.usefixtures("clean_cache")
def test_spmd_branch_does_not_use_cache(spmd_comm: Communicator) -> None:
    """``"spmd"`` returns a fresh engine each call — no cache leakage."""
    e1 = build_streaming_engine(_SPMD_PARAM, spmd_comm=spmd_comm)
    e2 = build_streaming_engine(_SPMD_PARAM, spmd_comm=spmd_comm)
    try:
        assert e1 is not e2
    finally:
        e1.shutdown()
        e2.shutdown()
