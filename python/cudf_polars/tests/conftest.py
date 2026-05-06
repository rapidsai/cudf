# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib.util
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
    # Do not use `pytest.importorskip` here: this fixture is autouse, so an
    # import-based skip would skip every test in the suite on environments
    # without rapidsmpf (e.g. the coverage CI job), masking real coverage.
    # We only want to gate the nranks>1 check on rapidsmpf being available.
    if importlib.util.find_spec("rapidsmpf") is None:
        return

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
    pytest.importorskip("rapidsmpf")
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
    return pl.GPUEngine(raise_on_fail=True)


def pytest_addoption(parser):
    parser.addoption(
        "--executor",
        action="store",
        default="streaming",
        choices=("in-memory", "streaming"),
        help="Executor to use for GPUEngine.",
    )

    parser.addoption(
        "--runtime",
        action="store",
        default="tasks",
        choices=("tasks", "rapidsmpf"),
        help="Runtime to use for the 'streaming' executor.",
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
        "skip_on_streaming_engine(reason): skip the test for streaming "
        '``engine`` variants (e.g. ``"spmd"``, ``"spmd-small"``) while '
        "still letting the in-memory variant run. Use this to track features "
        "that have no multi-partition implementation",
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

    if config.getoption("--runtime") == "rapidsmpf":
        if config.getoption("--executor") == "in-memory":
            raise pytest.UsageError("Rapidsmpf runtime requires --executor='streaming'")

        if importlib.util.find_spec("rapidsmpf") is None:
            raise pytest.UsageError(
                "Rapidsmpf runtime requires the 'rapidsmpf' package"
            )

    cudf_polars.testing.asserts.DEFAULT_EXECUTOR = config.getoption("--executor")
    cudf_polars.testing.asserts.DEFAULT_RUNTIME = config.getoption("--runtime")
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
        engine_param = callspec.params.get("_all_engine_param")
        if engine_param is None or engine_param == "in-memory":
            continue
        reason = (
            marker.args[0]
            if marker.args
            else marker.kwargs.get("reason", "unsupported on streaming engine")
        )
        item.add_marker(pytest.mark.skip(reason=reason))
