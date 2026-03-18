# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib.util
import os
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import pytest

import polars as pl

import cudf_polars.callback
from cudf_polars.utils.config import StreamingFallbackMode

if TYPE_CHECKING:
    from collections.abc import Generator

    import ray.runtime_context
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


@pytest.fixture(scope="session")
def ray_cluster() -> Generator[ray.runtime_context.RuntimeContext, None, None]:
    """Create one Ray cluster shared across the test session."""
    pytest.importorskip("ray")
    import ray

    # Ray raises a FutureWarning about overriding CUDA_VISIBLE_DEVICES.
    # Set `RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO` to silence it.
    os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"
    ray_context = ray.init(include_dashboard=False)
    with ray_context:
        yield ray_context


# Streaming engine variants. The `-small` suffix forces tiny partitions and
# silent fallback so multi-partition code paths get exercised even on tests
# with toy inputs.
_STREAMING_ENGINE_LABELS: list[str] = []
if importlib.util.find_spec("rapidsmpf") is not None:
    _STREAMING_ENGINE_LABELS += ["spmd", "spmd-small"]
    if importlib.util.find_spec("ray") is not None:
        _STREAMING_ENGINE_LABELS += ["ray", "ray-small"]

# All GPU engines.
_ENGINE_LABELS: list[str] = ["in-memory", *_STREAMING_ENGINE_LABELS]


@contextmanager
def _build_engine(
    label: str,
    request: pytest.FixtureRequest,
    options: dict[str, Any] | None = None,
) -> Generator[pl.GPUEngine, None, None]:
    """Construct a GPUEngine for the given backend label.

    ``label`` is one of ``"in-memory"``, ``"spmd"``, ``"spmd-small"``,
    ``"ray"``, ``"ray-small"``. The ``-small`` suffix forces tiny partitions
    and silent fallback. ``options`` is an optional dict with any of
    ``"executor_options"``, ``"engine_options"``, ``"rapidsmpf_options"`` —
    the calling fixture is responsible for sourcing it (typically from
    ``request.param`` when the fixture supports indirect parametrization).
    """
    if label == "in-memory":
        yield pl.GPUEngine(executor="in-memory", raise_on_fail=True)
        return

    from rapidsmpf.config import Options

    params: dict[str, Any] = options or {}
    executor_options: dict[str, Any] = {
        "max_rows_per_partition": 50,
        "dynamic_planning": {},
        "target_partition_size": 1_000_000,
    }
    if label.endswith("-small"):
        executor_options.update(
            max_rows_per_partition=4,
            target_partition_size=10,
            # We expect many tests to fall back, so silence the warnings.
            fallback_mode=StreamingFallbackMode.SILENT,
        )
    executor_options.update(params.get("executor_options", {}))
    rapidsmpf_options = (
        Options(params["rapidsmpf_options"]) if "rapidsmpf_options" in params else None
    )
    engine_options: dict[str, Any] = {
        "raise_on_fail": True,
        **params.get("engine_options", {}),
    }

    if label in ("spmd", "spmd-small"):
        from cudf_polars.experimental.rapidsmpf.frontend.spmd import SPMDEngine

        with SPMDEngine(
            comm=request.getfixturevalue("spmd_comm"),
            rapidsmpf_options=rapidsmpf_options,
            executor_options=executor_options,
            engine_options=engine_options,
        ) as engine:
            yield engine
    else:  # "ray" or "ray-small"
        from cudf_polars.experimental.rapidsmpf.frontend.ray import RayEngine

        ray_cluster = request.getfixturevalue("ray_cluster")
        # Use ``force_num_ranks`` so tests don't depend on the Ray cluster
        # exposing GPUs (which the session-scoped ``ray_cluster`` fixture does
        # not guarantee, and which would also conflict with the ``spmd``
        # variant grabbing the same physical GPU). Requires
        # ``allow_gpu_sharing=True``.
        with RayEngine(
            rapidsmpf_options=rapidsmpf_options,
            executor_options=executor_options,
            engine_options={"allow_gpu_sharing": True, **engine_options},
            ray_init_options={"address": "auto"},
            force_num_ranks=1,
        ) as engine:
            yield engine


@pytest.fixture(params=_STREAMING_ENGINE_LABELS)
def _streaming_engine_labels(request: pytest.FixtureRequest) -> str:
    """Internal: iterate over available streaming backends."""
    return request.param


@pytest.fixture
def streaming_engine(
    request: pytest.FixtureRequest,
    _streaming_engine_labels: str,
) -> Generator[StreamingEngine, None, None]:
    from cudf_polars.experimental.rapidsmpf.frontend.core import StreamingEngine

    options = getattr(request, "param", None) or {}
    with _build_engine(_streaming_engine_labels, request, options) as engine:
        assert isinstance(engine, StreamingEngine)
        yield engine


@pytest.fixture(params=["spmd", "spmd-small"])
def spmd_engine(
    request: pytest.FixtureRequest,
) -> Generator[StreamingEngine, None, None]:
    """Like :func:`streaming_engine`, but only over SPMD backends.

    Use for tests that depend on SPMD-specific APIs (e.g. ``engine.comm`` /
    ``engine.context``) which :class:`RayEngine` does not expose.
    """
    from cudf_polars.experimental.rapidsmpf.frontend.spmd import SPMDEngine

    with _build_engine(request.param, request) as engine:
        assert isinstance(engine, SPMDEngine)
        yield engine


@pytest.fixture(params=_ENGINE_LABELS)
def _all_engine_labels(request: pytest.FixtureRequest) -> str:
    """Internal: iterate over all available backends (streaming or non-streaming)."""
    return request.param


@pytest.fixture
def engine(
    request: pytest.FixtureRequest,
    _all_engine_labels: str,
) -> Generator[pl.GPUEngine, None, None]:
    options = getattr(request, "param", None) or {}
    with _build_engine(_all_engine_labels, request, options) as engine:
        yield engine


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
    """Apply ``skip_on_streaming_engine`` markers to streaming ``_all_engine_labels`` items."""
    for item in items:
        marker = item.get_closest_marker("skip_on_streaming_engine")
        if marker is None:
            continue
        callspec = getattr(item, "callspec", None)
        if callspec is None:
            continue
        label = callspec.params.get("_all_engine_labels")
        if label is None or label == "in-memory":
            continue
        reason = (
            marker.args[0]
            if marker.args
            else marker.kwargs.get("reason", "unsupported on streaming engine")
        )
        item.add_marker(pytest.mark.skip(reason=reason))
