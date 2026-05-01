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
    shutdown_streaming_engine_cache,
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


@pytest.fixture(scope="session")
def ray_cluster() -> Generator[None, None, None]:
    """Session-scoped Ray cluster — one ``ray.init`` shared by all ray tests.

    Skipped if ``ray`` is not installed or the suite is running under
    ``rrun`` (``RayEngine`` refuses to run inside rrun).

    Yields nothing — this fixture is a side-effect setup gate. The cached
    :class:`RayEngine` (built by ``build_streaming_engine``) lives until
    this fixture's teardown runs ``shutdown_streaming_engine_cache``
    *before* ``ray.shutdown()`` so the actor handles aren't invalidated
    against a torn-down cluster.
    """
    pytest.importorskip("ray")
    import os
    import tempfile

    import ray
    from rapidsmpf.bootstrap import is_running_with_rrun

    if is_running_with_rrun():
        pytest.skip("RayEngine cannot be created inside an rrun cluster")

    if not ray.is_initialized():
        # Prevent Ray from overriding ``CUDA_VISIBLE_DEVICES`` to ``""``
        # when a worker process starts with zero visible GPUs (e.g. the
        # test driver itself). Future Ray versions error if this isn't
        # set explicitly. Must be set *before* ``ray.init``.
        os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")
        # Per-process temp dir disables ``/tmp/ray/session_latest``-based
        # auto-discovery, so concurrent pytest runs (xdist, parallel CI
        # invocations, repeated local runs) don't accidentally connect
        # to each other's clusters.
        temp_dir = tempfile.mkdtemp(prefix=f"ray-pytest-{os.getpid()}-")
        ray.init(include_dashboard=False, _temp_dir=temp_dir)
    try:
        yield
    finally:
        # Release the cached RayEngine before tearing down the cluster.
        shutdown_streaming_engine_cache()
        ray.shutdown()


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
    request: pytest.FixtureRequest,
    _streaming_engine_param: EngineFixtureParam,
    spmd_comm: Communicator,
) -> Generator[Callable[..., StreamingEngine], None, None]:
    """
    Yield a factory that constructs :class:`StreamingEngine` instances for tests.

    The fixture is parametrized over :data:`STREAMING_ENGINE_FIXTURE_PARAMS`.
    SPMD-backend engines are created fresh per call and shut down at test
    end. Ray-backend engines come from the session-scoped cache inside
    :func:`build_streaming_engine` (via :meth:`RayEngine._reset`); the cache
    owns their lifetime, so the per-test teardown skips ray engines.

    Parameters
    ----------
    request
        Pytest fixture request used to lazily resolve ``ray_cluster`` only
        for ray-parametrized variants.
    _streaming_engine_param
        Parametrized engine descriptor controlling backend and block size mode.
    spmd_comm
        Communicator used when constructing SPMD-based engines.

    Yields
    ------
    Factory function that creates :class:`StreamingEngine` instances. The
    factory accepts optional :class:`StreamingOptions`, which are merged on
    top of the parametrized blocksize baseline.
    """
    if _streaming_engine_param.engine_name == "ray":
        request.getfixturevalue("ray_cluster")

    engines: list[StreamingEngine] = []

    def factory(options: StreamingOptions | None = None) -> StreamingEngine:
        engine = build_streaming_engine(_streaming_engine_param, spmd_comm, options)
        engines.append(engine)
        return engine

    yield factory

    # RayEngine instances are owned by the session-level cache (one
    # cached engine reused via ``_reset`` across all ray tests); only
    # shut down per-call engines for non-ray backends.
    if _streaming_engine_param.engine_name != "ray":
        for engine in reversed(engines):
            engine.shutdown()


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
) -> Generator[pl.GPUEngine, None, None]:
    """
    Yield a :class:`polars.GPUEngine` for each engine variant under test.

    Parameters
    ----------
    request
        Pytest fixture request object used to access dependent fixtures.
    _all_engine_param
        Parametrized engine descriptor covering both in-memory and streaming
        variants.

    Yields
    ------
    Engine instance matching the parametrized variant.

    Notes
    -----
    For tests that require a :class:`StreamingEngine` only, use the
    :func:`streaming_engine` fixture instead.
    """
    if _all_engine_param.engine_name == "in-memory":
        yield pl.GPUEngine(executor="in-memory", raise_on_fail=True)
        return

    if _all_engine_param.engine_name == "ray":
        request.getfixturevalue("ray_cluster")

    spmd_comm: Communicator = request.getfixturevalue("spmd_comm")
    engine = build_streaming_engine(_all_engine_param, spmd_comm)
    try:
        yield engine
    finally:
        # RayEngine instances are owned by the session-level cache;
        # SPMDEngine instances are owned by this fixture call.
        if _all_engine_param.engine_name != "ray":
            engine.shutdown()


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
        choices=("single", "distributed"),
        help="Cluster to use for 'streaming' executor.",
    )


def pytest_configure(config):
    import cudf_polars.testing.asserts

    config.addinivalue_line(
        "markers",
        "skip_on_streaming_engine(reason, *, backend=None): skip the test "
        "for streaming ``engine`` variants while still letting the in-memory "
        "variant run. Use this to track features that have no "
        "multi-partition implementation. By default, all streaming variants "
        '(e.g. ``"spmd"``, ``"spmd-small"``, ``"ray"``) are skipped; pass '
        '``backend="ray"`` (a single backend name) or '
        '``backend=("ray", ...)`` (a sequence of names) to skip only '
        "those backends.",
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
    """Apply ``skip_on_streaming_engine`` markers during test collection.

    The ``skip_on_streaming_engine`` marker supports an optional
    ``backend=`` keyword:

    - If ``backend`` is ``None`` (default), the test is skipped for all
      streaming engine variants.
    - If ``backend`` is specified (e.g. ``"ray"`` or ``("ray",)``), the
      test is skipped only for matching variants, including suffixed
      forms such as ``"ray-small"``.

    The skip reason is taken from the first positional argument if
    provided, otherwise from the ``reason=`` keyword, and defaults to
    ``"unsupported on streaming engine"``.
    """
    for item in items:
        marker = item.get_closest_marker("skip_on_streaming_engine")
        if marker is None:
            continue
        callspec = getattr(item, "callspec", None)
        if callspec is None:
            continue
        # Cover both fixture paths: ``_all_engine_param`` for the
        # ``engine`` fixture, ``_streaming_engine_param`` for the
        # ``streaming_engine`` / ``streaming_engine_factory`` fixtures.
        engine_param = callspec.params.get("_all_engine_param") or callspec.params.get(
            "_streaming_engine_param"
        )
        if engine_param is None or engine_param == "in-memory":
            continue
        backend_filter = marker.kwargs.get("backend")
        if backend_filter is not None:
            engine_backend = engine_param.removesuffix("-small")
            backends = (
                (backend_filter,)
                if isinstance(backend_filter, str)
                else tuple(backend_filter)
            )
            if engine_backend not in backends:
                continue
        reason = (
            marker.args[0]
            if marker.args
            else marker.kwargs.get("reason", "unsupported on streaming engine")
        )
        item.add_marker(pytest.mark.skip(reason=reason))
