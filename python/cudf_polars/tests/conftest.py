# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING, Any, Literal

import pytest

import polars as pl

import cudf_polars.callback
from cudf_polars.utils.config import StreamingFallbackMode

if TYPE_CHECKING:
    from collections.abc import Generator

    from rapidsmpf.communicator.communicator import Communicator

    from cudf_polars.experimental.rapidsmpf.frontend.core import StreamingEngine


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


@pytest.fixture
def using_streaming_engine(engine: pl.GPUEngine) -> bool:
    """True when the active ``engine`` fixture is a :class:`StreamingEngine`."""
    try:
        from cudf_polars.experimental.rapidsmpf.frontend.core import StreamingEngine

        return isinstance(engine, StreamingEngine)
    except ImportError:
        return False


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


@pytest.fixture
def blocksize_mode(request: pytest.FixtureRequest) -> Literal["default", "small"]:
    """Blocksize mode for the streaming executor.

    Defaults to ``"default"``. Tests can override this via ``indirect``
    parametrization with ``["default", "small"]`` to run under both the
    standard and small-partition configurations. In addition, the
    ``engine="spmd-small"`` variant of the ``engine`` fixture implicitly
    selects ``"small"`` mode so that every streaming-engine test exercises
    tiny-partition / fallback paths without per-test opt-in. Explicit
    indirect parametrization always wins over the implicit engine-derived
    value.
    """
    if hasattr(request, "param"):
        return request.param
    callspec = getattr(request.node, "callspec", None)
    if callspec is not None and callspec.params.get("engine") == "spmd-small":
        return "small"
    return "default"


@pytest.fixture
def streaming_engine(
    request: pytest.FixtureRequest,
    spmd_comm: Communicator,
    blocksize_mode: Literal["default", "small"],
) -> Generator[StreamingEngine, None, None]:
    """Yield an :class:`SPMDEngine` configured for streaming-only tests.

    Options can be overridden via ``indirect`` parametrization by passing
    a dict with any of the keys ``"executor_options"``,
    ``"engine_options"``, or ``"rapidsmpf_options"``.
    """
    from rapidsmpf.config import Options

    from cudf_polars.experimental.rapidsmpf.frontend.spmd import SPMDEngine

    params: dict[str, Any] = getattr(request, "param", {}) or {}
    executor_options: dict[str, Any] = {
        "max_rows_per_partition": 50,
        "dynamic_planning": {},
        "target_partition_size": 1_000_000,
    }
    if blocksize_mode == "small":
        executor_options.update(
            max_rows_per_partition=4,
            target_partition_size=10,
            # We expect many tests to fall back, so silence the warnings
            fallback_mode=StreamingFallbackMode.SILENT,
        )
    executor_options.update(params.get("executor_options", {}))
    rapidsmpf_options = (
        Options(params.get("rapidsmpf_options"))
        if "rapidsmpf_options" in params
        else None
    )
    engine_options: dict[str, Any] = {"raise_on_fail": True}
    engine_options.update(params.get("engine_options", {}))
    with SPMDEngine(
        comm=spmd_comm,
        rapidsmpf_options=rapidsmpf_options,
        executor_options=executor_options,
        engine_options=engine_options,
    ) as engine:
        yield engine


_ENGINE_PARAMS = ["in-memory"]
if importlib.util.find_spec("rapidsmpf") is not None:
    _ENGINE_PARAMS.extend(["spmd", "spmd-small"])


@pytest.fixture(params=_ENGINE_PARAMS)
def engine(
    request: pytest.FixtureRequest,
) -> Generator[pl.GPUEngine, None, None]:
    """Yield a :class:`polars.GPUEngine` for each engine variant under test.

    Use this fixture for tests that support any ``GPUEngine``. The test runs
    once per available variant: always with the in-memory executor, and, if
    ``rapidsmpf`` is installed, a streaming :class:`SPMDEngine` at the
    default blocksize (``"spmd"``) and a second streaming engine with
    tiny-partition / silent-fallback settings (``"spmd-small"``). The
    ``"spmd-small"`` variant forces ``blocksize_mode="small"`` via the
    ``blocksize_mode`` fixture, so every test using ``engine`` exercises
    multi-partition paths for free.

    For tests that require a ``StreamingEngine``, use the ``streaming_engine``
    fixture instead.
    """
    if request.param == "in-memory":
        yield pl.GPUEngine(executor="in-memory", raise_on_fail=True)
    else:
        yield request.getfixturevalue("streaming_engine")


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
        "skip_on_streaming_engine(reason): skip the test when the `engine` "
        "fixture resolves to a streaming engine variant (e.g. 'spmd'). "
        "Use for tests exercising operations that have no multi-partition "
        "implementation.",
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
    """Apply ``skip_on_streaming_engine`` markers to parametrized ``engine`` items."""
    for item in items:
        marker = item.get_closest_marker("skip_on_streaming_engine")
        if marker is None:
            continue
        callspec = getattr(item, "callspec", None)
        if callspec is None:
            continue
        engine_param = callspec.params.get("engine")
        if engine_param is None or engine_param == "in-memory":
            continue
        reason = (
            marker.args[0]
            if marker.args
            else marker.kwargs.get("reason", "unsupported on streaming engine")
        )
        item.add_marker(pytest.mark.skip(reason=reason))
