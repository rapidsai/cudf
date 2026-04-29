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
    from collections.abc import Callable, Generator

    from rapidsmpf.communicator.communicator import Communicator

    from cudf_polars.experimental.rapidsmpf.frontend.core import StreamingEngine
    from cudf_polars.experimental.rapidsmpf.frontend.options import StreamingOptions


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
def spmd_comm() -> Communicator:
    """Session-scoped communicator — bootstrapped once and shared across all tests.

    Sharing a single communicator avoids the file-based bootstrap race that can
    cause hangs when ``create_ucxx_comm()`` is called repeatedly in the same
    ``rrun`` session (stale barrier files / stale ``ucxx_root_address`` KV entry).
    """
    pytest.importorskip("rapidsmpf")
    from rapidsmpf import bootstrap
    from rapidsmpf.communicator.single import new_communicator as single_communicator
    from rapidsmpf.config import Options, get_environment_variables
    from rapidsmpf.progress_thread import ProgressThread

    if bootstrap.is_running_with_rrun():
        return bootstrap.create_ucxx_comm(
            progress_thread=ProgressThread(),
            type=bootstrap.BackendType.AUTO,
        )
    return single_communicator(Options(get_environment_variables()), ProgressThread())


@pytest.fixture(params=STREAMING_ENGINE_FIXTURE_PARAMS)
def _streaming_engine_param(request: pytest.FixtureRequest) -> EngineFixtureParam:
    return EngineFixtureParam(full_name=request.param)


@pytest.fixture(params=ALL_ENGINE_FIXTURE_PARAMS)
def _all_engine_param(request: pytest.FixtureRequest) -> EngineFixtureParam:
    return EngineFixtureParam(full_name=request.param)


@pytest.fixture
def streaming_engine_factory(
    _streaming_engine_param: EngineFixtureParam,
    spmd_comm: Communicator,
) -> Generator[Callable[..., StreamingEngine], None, None]:
    """Yield a callable that constructs and manages :class:`StreamingEngine` lifetimes.

    Parametrized over :data:`STREAMING_ENGINE_FIXTURE_PARAMS` (today: ``"spmd"`` and
    ``"spmd-small"``). Tests pass a :class:`StreamingOptions` describing only
    the fields they care about; the factory layers the parametrized
    blocksize baseline underneath and tracks engines for teardown.

    Today the factory builds an :class:`SPMDEngine`, but it is typed as the
    abstract :class:`StreamingEngine` because future variants (``RayEngine``,
    ``DaskEngine``) will plug in here. Tests that genuinely need
    SPMDEngine-specific attributes (``comm``, ``context``) should construct
    an :class:`SPMDEngine` directly using the ``spmd_comm`` fixture.
    """
    engines: list[StreamingEngine] = []

    def factory(options: StreamingOptions | None = None) -> StreamingEngine:
        engine = build_streaming_engine(_streaming_engine_param, spmd_comm, options)
        engines.append(engine)
        return engine

    yield factory

    for engine in reversed(engines):
        engine.shutdown()


@pytest.fixture
def streaming_engine(
    streaming_engine_factory: Callable[..., StreamingEngine],
) -> StreamingEngine:
    """Default-configured :class:`StreamingEngine` (no per-test overrides).

    Inherits :func:`streaming_engine_factory`'s parametrization, so tests
    using this fixture run once per ``(backend, blocksize_mode)``
    combination.
    """
    return streaming_engine_factory()


@pytest.fixture
def engine(
    request: pytest.FixtureRequest,
    _all_engine_param: EngineFixtureParam,
) -> Generator[pl.GPUEngine, None, None]:
    """Yield a :class:`polars.GPUEngine` for each engine variant under test.

    Iterates over ``"in-memory"`` and (when ``rapidsmpf`` is installed) the
    streaming variants from :data:`STREAMING_ENGINE_FIXTURE_PARAMS` (``"spmd"`` and
    ``"spmd-small"``). Streaming engines are built inline via
    :func:`build_streaming_engine` so that ``engine="in-memory"`` does not
    pull in :func:`spmd_comm` (which would skip on rapidsmpf-less envs).

    For tests that need a :class:`StreamingEngine` only, use the
    :func:`streaming_engine` fixture instead.
    """
    if _all_engine_param.engine_name == "in-memory":
        yield pl.GPUEngine(executor="in-memory", raise_on_fail=True)
        return

    spmd_comm: Communicator = request.getfixturevalue("spmd_comm")
    engine = build_streaming_engine(_all_engine_param, spmd_comm)
    try:
        yield engine
    finally:
        engine.shutdown()


@pytest.fixture
def engine_raise_on_fail() -> pl.GPUEngine:
    """Yield a default :class:`polars.GPUEngine` with ``raise_on_fail=True``.

    Intended for error-path tests that assert specific exceptions propagate
    from ``.collect()``. Uses the default (in-memory) executor so errors are
    not wrapped by a streaming task group.
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
        choices=("single", "distributed"),
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

    if (
        config.getoption("--cluster") == "distributed"
        and config.getoption("--executor") != "streaming"
    ):
        raise pytest.UsageError("Distributed cluster requires --executor='streaming'")

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
