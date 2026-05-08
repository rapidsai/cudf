# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import polars as pl

import cudf_polars.callback
from cudf_polars.testing.engine_utils import (
    ALL_ENGINE_FIXTURE_PARAMS,
    STREAMING_ENGINE_FIXTURE_PARAMS,
    EngineFixtureParam,
    build_streaming_engine,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Mapping
    from typing import TypeAlias

    from cudf_polars.experimental.rapidsmpf.frontend.core import StreamingEngine
    from cudf_polars.experimental.rapidsmpf.frontend.options import StreamingOptions
    from cudf_polars.experimental.rapidsmpf.frontend.spmd import SPMDEngine

    # Read-only view over the per-backend streaming engines owned by the
    # ``streaming_engines`` session fixture. Only that fixture mutates the
    # underlying dict; consumers (``spmd_engine``, ``streaming_engine_factory``,
    # ``engine``) only look up by backend name.
    StreamingEngines: TypeAlias = Mapping[str, StreamingEngine]


# Number of ranks for multi-rank streaming engines that share one GPU
# (currently ``RayEngine``). Single-GPU dev hosts and CI runners require
# ``allow_gpu_sharing=True`` to oversubscribe one device across actors.
NUM_RANKS = 2


@pytest.fixture(params=[False, True], ids=["no_nulls", "nulls"], scope="session")
def with_nulls(request):
    return request.param


@pytest.fixture
def clear_memory_resource_cache():
    """
    Clear the cudf_polars.callback.default_memory_resource cache before and after a test.

    This function caches memory resources for the duration of the process. Any test that
    creates a pool (e.g. ``CudaAsyncMemoryResource``) should use this fixture to ensure that
    the pool is freed after the test.
    """
    cudf_polars.callback.default_memory_resource.cache_clear()
    yield
    cudf_polars.callback.default_memory_resource.cache_clear()


@pytest.fixture(autouse=True)
def _skip_unless_spmd(request: pytest.FixtureRequest) -> None:
    """Skip tests in SPMD multi-rank mode unless marked with ``pytest.mark.spmd``."""
    from rapidsmpf.bootstrap import get_nranks, is_running_with_rrun

    if (
        is_running_with_rrun()
        and get_nranks() > 1
        and not request.node.get_closest_marker("spmd")
    ):
        pytest.skip("skip: SPMD nranks > 1 (mark with pytest.mark.spmd to run)")


@pytest.fixture(scope="session")
def streaming_engines() -> Generator[StreamingEngines, None, None]:
    """Return a session-scoped mapping of engine name to engine instance.

    The returned :class:`StreamingEngines` is a dict that maps each engine
    name to a single shared engine instance, which is reused across the entire
    test session.
    """
    from rapidsmpf import bootstrap
    from rapidsmpf.communicator.single import new_communicator as single_communicator
    from rapidsmpf.config import Options, get_environment_variables
    from rapidsmpf.progress_thread import ProgressThread

    from cudf_polars.experimental.rapidsmpf.frontend.spmd import SPMDEngine

    if bootstrap.is_running_with_rrun():
        comm = bootstrap.create_ucxx_comm(
            progress_thread=ProgressThread(),
            type=bootstrap.BackendType.AUTO,
        )
    else:
        comm = single_communicator(
            Options(get_environment_variables()), ProgressThread()
        )

    engines: dict[str, StreamingEngine] = {"spmd": SPMDEngine(comm=comm)}

    if "dask" in STREAMING_ENGINE_FIXTURE_PARAMS:  # pragma: no cover
        from cudf_polars.experimental.rapidsmpf.frontend.dask import DaskEngine

        engines["dask"] = DaskEngine(engine_options={"allow_gpu_sharing": True})

    if "ray" in STREAMING_ENGINE_FIXTURE_PARAMS:  # pragma: no cover
        from cudf_polars.experimental.rapidsmpf.frontend.ray import RayEngine

        # Always pin ``num_ranks`` so the cached engine has a deterministic
        # actor count regardless of how many GPUs the host happens to have;
        # otherwise ``RayEngine`` defaults to ``get_num_gpus_in_ray_cluster()``
        # and tests that depend on rank-count behavior (e.g. fast-count
        # parquet, concat) become non-portable. Pinning ``num_ranks`` requires
        # ``allow_gpu_sharing=True`` (production guard).
        engines["ray"] = RayEngine(
            num_ranks=NUM_RANKS,
            engine_options={"allow_gpu_sharing": True},
            ray_init_options={"include_dashboard": False},
        )

    try:
        yield engines
    finally:
        while engines:
            _, engine = engines.popitem()
            engine.shutdown()


@pytest.fixture
def spmd_engine(streaming_engines: StreamingEngines) -> SPMDEngine:
    """Return the shared :class:`SPMDEngine` reset to default options."""
    from cudf_polars.experimental.rapidsmpf.frontend.spmd import SPMDEngine

    engine = streaming_engines["spmd"]
    assert isinstance(engine, SPMDEngine)
    engine._reset()
    return engine


@pytest.fixture
def spmd_engine_factory(
    streaming_engines: StreamingEngines,
) -> Callable[..., SPMDEngine]:
    """
    Return a factory that yields the shared :class:`SPMDEngine`.

    Use this in place of :func:`streaming_engine_factory` for tests that
    must run on SPMD only.
    """
    from cudf_polars.experimental.rapidsmpf.frontend.spmd import SPMDEngine

    param = EngineFixtureParam(full_name="spmd")

    def factory(options: StreamingOptions | None = None) -> SPMDEngine:
        engine = build_streaming_engine(param, streaming_engines, options)
        assert isinstance(engine, SPMDEngine)
        return engine

    return factory


@pytest.fixture(params=STREAMING_ENGINE_FIXTURE_PARAMS)
def _streaming_engine_param(request: pytest.FixtureRequest) -> EngineFixtureParam:
    """Parametrization helper to run tests for each streaming engine variant."""
    return EngineFixtureParam(full_name=request.param)


@pytest.fixture(params=ALL_ENGINE_FIXTURE_PARAMS)
def _all_engine_param(request: pytest.FixtureRequest) -> EngineFixtureParam:
    """Parametrization helper to run tests for each engine variant."""
    return EngineFixtureParam(full_name=request.param)


@pytest.fixture
def streaming_engine_factory(
    _streaming_engine_param: EngineFixtureParam,
    streaming_engines: StreamingEngines,
) -> Callable[..., StreamingEngine]:
    """
    Return a factory that yields a shared :class:`StreamingEngine`.

    Parameters
    ----------
    _streaming_engine_param
        Parametrized engine descriptor controlling backend and block size mode.
    streaming_engines
        Session-scoped engine collection to look up the shared engine in.

    Returns
    -------
    Factory function that returns the shared :class:`StreamingEngine`.
    """

    def factory(options: StreamingOptions | None = None) -> StreamingEngine:
        return build_streaming_engine(
            _streaming_engine_param, streaming_engines, options
        )

    return factory


@pytest.fixture
def streaming_engine(
    streaming_engine_factory: Callable[..., StreamingEngine],
) -> StreamingEngine:
    """
    Return a default-configured :class:`StreamingEngine`.

    Inherits the parametrization of :func:`streaming_engine_factory`, so
    tests using this fixture run once per ``(backend, blocksize_mode)``
    combination.

    Parameters
    ----------
    streaming_engine_factory
        Factory fixture used to construct streaming engines.

    Returns
    -------
    A streaming engine created with the parametrized baseline and no
    per-test overrides.
    """
    return streaming_engine_factory()


@pytest.fixture
def engine(
    request: pytest.FixtureRequest,
    _all_engine_param: EngineFixtureParam,
) -> pl.GPUEngine:
    """
    Return a :class:`polars.GPUEngine` for each engine variant under test.

    Parameters
    ----------
    request
        Pytest fixture request object used to access dependent fixtures.
    _all_engine_param
        Parametrized engine descriptor covering both in-memory and streaming
        variants.

    Returns
    -------
    Engine instance matching the parametrized variant.

    Notes
    -----
    For tests that require a :class:`StreamingEngine` only, use the
    :func:`streaming_engine` fixture instead.
    """
    if _all_engine_param.engine_name == "in-memory":
        return pl.GPUEngine(executor="in-memory", raise_on_fail=True)

    engines: StreamingEngines = request.getfixturevalue("streaming_engines")
    return build_streaming_engine(_all_engine_param, engines)


@pytest.fixture
def engine_raise_on_fail() -> pl.GPUEngine:
    """
    Return a default :class:`polars.GPUEngine` with ``raise_on_fail=True``.

    Returns
    -------
    In-memory engine configured to raise exceptions on failure.

    Notes
    -----
    Intended for error-path tests that assert specific exceptions propagate
    from ``.collect()``. Uses the in-memory executor so errors are not wrapped
    by a streaming task group.
    """
    # TODO: We should be testing will all supported engine variants
    return pl.GPUEngine(executor="in-memory", raise_on_fail=True)


def pytest_addoption(parser):
    parser.addoption(
        "--executor",
        action="store",
        default="streaming",
        choices=("in-memory", "streaming"),
        help="Executor to use for GPUEngine.",
    )

    parser.addoption(
        "--cluster",
        action="store",
        default="single",
        choices=("single",),
        help="Cluster to use for 'streaming' executor.",
    )


def pytest_configure(config):
    import cudf_polars.testing.asserts

    config.addinivalue_line(
        "markers",
        "skip_on_streaming_engine(reason, *, engine=None): skip the test for "
        'streaming ``engine`` variants (e.g. ``"spmd"``, ``"spmd-small"``, '
        '``"dask"``, ``"ray"``) while still allowing the in-memory variant to run.',
    )

    # Ray's internal subprocess management leaks `/dev/null` file handles, and
    # distributed's shutdown leaves unclosed sockets. Under Python 3.14 +
    # pytest 9, these surface as unraisable `ResourceWarning`s and — combined
    # with `filterwarnings = ["error", ...]` in pyproject.toml — fail
    # otherwise-unrelated tests when the GC finalizer happens to fire during
    # them. With `pytest-xdist --dist=worksteal`, the leak can land in any
    # test that shares a worker with a ray/dask test, so the suppression must
    # apply globally rather than per-module.
    config.addinivalue_line("filterwarnings", "ignore::ResourceWarning")

    cudf_polars.testing.asserts.DEFAULT_EXECUTOR = config.getoption("--executor")
    cudf_polars.testing.asserts.DEFAULT_CLUSTER = config.getoption("--cluster")


def pytest_collection_modifyitems(items):
    """Apply ``skip_on_streaming_engine`` markers to streaming ``engine`` items."""
    for item in items:
        marker = item.get_closest_marker("skip_on_streaming_engine")
        if marker is None:
            continue
        callspec = getattr(item, "callspec", None)
        if callspec is None:
            continue
        # Tests bind to either ``engine`` (parametrized via ``_all_engine_param``)
        # or ``streaming_engine`` / ``streaming_engine_factory`` (parametrized via
        # ``_streaming_engine_param``). Check both.
        engine_param = callspec.params.get("_all_engine_param") or callspec.params.get(
            "_streaming_engine_param"
        )
        if engine_param is None or engine_param == "in-memory":
            continue
        engine_filter = marker.kwargs.get("engine")
        if engine_filter is not None:
            if isinstance(engine_filter, str):
                engine_filter = (engine_filter,)
            # Strip the ``-small`` suffix so ``"spmd-small"`` matches
            # ``engine=("spmd",)``.
            engine_name = engine_param.removesuffix("-small")
            if engine_name not in engine_filter:
                continue
        reason = (
            marker.args[0]
            if marker.args
            else marker.kwargs.get("reason", "unsupported on streaming engine")
        )
        item.add_marker(pytest.mark.skip(reason=reason))
