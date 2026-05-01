# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for :class:`polars.GPUEngine` tests."""

from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from rapidsmpf.communicator.communicator import Communicator

    import polars as pl

    from cudf_polars.experimental.rapidsmpf.frontend.core import StreamingEngine
    from cudf_polars.experimental.rapidsmpf.frontend.options import StreamingOptions


STREAMING_ENGINE_FIXTURE_PARAMS: list[str] = []
if importlib.util.find_spec("rapidsmpf") is not None:
    STREAMING_ENGINE_FIXTURE_PARAMS.extend(["spmd", "spmd-small"])
    # ``"ray"`` is gated additionally on ``ray`` being installed and not
    # running under ``rrun`` (``RayEngine`` raises if it is).
    if importlib.util.find_spec("ray") is not None:  # pragma: no cover
        from rapidsmpf.bootstrap import is_running_with_rrun as _is_running_with_rrun

        if not _is_running_with_rrun():
            STREAMING_ENGINE_FIXTURE_PARAMS.append("ray")
        del _is_running_with_rrun
ALL_ENGINE_FIXTURE_PARAMS = ["in-memory", *STREAMING_ENGINE_FIXTURE_PARAMS]

# Default rank count for test-side ``RayEngine`` constructions. Multi-
# rank exercises the cross-rank UCXX barrier inside
# :meth:`RankActor.reset` and the multi-actor evaluate path. All
# test-side ``RayEngine(...)`` calls should pass ``num_ranks=NUM_RANKS``
# (paired with ``engine_options={"allow_gpu_sharing": True}``) so they
# coexist on a single GPU with the cached engine in
# :func:`build_streaming_engine`.
NUM_RANKS: int = 1


@dataclass
class EngineFixtureParam:
    """
    Decoded engine parametrization identifier.

    Attributes
    ----------
    full_name
        Full parametrization id e.g. "in-memory", "spmd", or "spmd-small".
    engine_name
        Backend name (e.g. ``"in-memory"`` or ``"spmd"``).
    blocksize_mode
        Block size mode, either ``"medium"`` or ``"small"``.
    """

    full_name: str
    engine_name: str
    blocksize_mode: Literal["medium", "small"]

    def __init__(self, full_name: str):
        self.full_name = full_name
        if full_name.endswith("-small"):
            self.engine_name = full_name.removesuffix("-small")
            self.blocksize_mode = "small"
        else:
            # Covers ``"in-memory"`` and bare backend names like ``"spmd"``.
            self.engine_name = full_name
            self.blocksize_mode = "medium"


def is_streaming_engine(obj: Any) -> bool:
    """Return ``True`` if ``obj`` is a :class:`StreamingEngine`."""
    try:
        from cudf_polars.experimental.rapidsmpf.frontend.core import StreamingEngine
    except ImportError:  # pragma: no cover; only triggered without rapidsmpf
        return False
    return isinstance(obj, StreamingEngine)


def create_streaming_options(
    blocksize_mode: Literal["medium", "small"],
    overrides: StreamingOptions | None = None,
) -> StreamingOptions:
    """
    Create :class:`StreamingOptions` for a block size mode.

    Parameters
    ----------
    blocksize_mode
        Block size configuration. ``"medium"`` uses moderate partition sizes,
        while ``"small"`` uses very small partitions and sets
        ``fallback_mode=SILENT`` to avoid excessive warnings from CPU fallback.
    overrides
        Optional options to merge on top of the selected baseline. Fields in
        ``overrides`` take precedence over the baseline.

    Returns
    -------
    The merged streaming options.
    """
    from cudf_polars.experimental.rapidsmpf.frontend.options import StreamingOptions
    from cudf_polars.utils.config import StreamingFallbackMode

    match blocksize_mode:
        case "medium":
            baseline = StreamingOptions(
                max_rows_per_partition=50,
                dynamic_planning={},
                target_partition_size=1_000_000,
                raise_on_fail=True,
            )
        case "small":
            baseline = StreamingOptions(
                max_rows_per_partition=4,
                dynamic_planning={},
                target_partition_size=10,
                raise_on_fail=True,
                fallback_mode=StreamingFallbackMode.SILENT,
            )
        case _:  # pragma: no cover
            raise ValueError(f"Unknown blocksize_mode: {blocksize_mode!r}")
    if overrides is None:
        return baseline
    return StreamingOptions(**{**baseline.to_dict(), **overrides.to_dict()})


# Single-slot module-level cache for the most recently built
# :class:`RayEngine`. Subsequent ``build_streaming_engine`` calls for the
# ``"ray"`` backend reuse this engine via :meth:`RayEngine._reset` instead
# of paying the full actor-fork + UCXX bootstrap cost again. Released by
# :func:`shutdown_streaming_engine_cache`, which test session-teardown
# fixtures must call before ``ray.shutdown()`` so the engine's actor
# handles aren't invalidated against a torn-down cluster.
_cached_ray_engine: StreamingEngine | None = None


def build_streaming_engine(
    param: EngineFixtureParam,
    spmd_comm: Communicator,
    options: StreamingOptions | None = None,
) -> StreamingEngine:
    """
    Build a :class:`StreamingEngine` from an engine fixture parameter.

    For ``param.engine_name == "ray"`` a single :class:`RayEngine` is
    cached at module scope and reused via :meth:`RayEngine._reset` on
    subsequent calls — this amortizes the multi-second actor-fork +
    UCXX-bootstrap cost across all tests in a session. Other backends
    are constructed fresh on every call.

    Parameters
    ----------
    param
        Decoded engine fixture parameter describing the backend and block size mode.
    spmd_comm
        Communicator used when constructing an :class:`SPMDEngine`. Unused
        for the ``"ray"`` branch.
    options
        Optional streaming options to merge on top of the baseline selected by
        ``param.blocksize_mode``.

    Returns
    -------
    A streaming engine matching ``param``.
    """
    streaming_options = create_streaming_options(param.blocksize_mode, options)
    common_kwargs: dict[str, Any] = {
        "rapidsmpf_options": streaming_options.to_rapidsmpf_options(),
        "executor_options": streaming_options.to_executor_options(),
        "engine_options": streaming_options.to_engine_options(),
    }
    match param.engine_name:
        case "spmd":
            from cudf_polars.experimental.rapidsmpf.frontend.spmd import SPMDEngine

            return SPMDEngine(comm=spmd_comm, **common_kwargs)
        case "ray":  # pragma: no cover
            return _build_or_reset_ray_engine(common_kwargs)
        case _:  # pragma: no cover
            raise AssertionError(f"Unknown streaming backend: {param.engine_name!r}")


def _build_or_reset_ray_engine(  # pragma: no cover
    common_kwargs: dict[str, Any],
) -> StreamingEngine:
    """
    Return the cached :class:`RayEngine`, ``_reset``-ed to ``common_kwargs``.

    Builds a fresh engine on the first call. On subsequent calls reuses
    the cached engine and applies ``common_kwargs`` via
    :meth:`RayEngine._reset` — preserving the rank actors and the UCXX
    communicator. ``allow_gpu_sharing=True`` is injected on every call
    because :class:`StreamingEngine` re-validates the
    ``num_ranks > 1`` invariant on every reset.
    """
    global _cached_ray_engine  # noqa: PLW0603
    # ``_reset`` replaces engine state in full, so ``allow_gpu_sharing``
    # must be re-asserted on every call to satisfy the
    # ``num_ranks > 1`` validation in ``StreamingEngine.__init__``.
    engine_options = {
        **common_kwargs.get("engine_options", {}),
        "allow_gpu_sharing": True,
    }
    common_kwargs = {**common_kwargs, "engine_options": engine_options}
    if _cached_ray_engine is None:
        from cudf_polars.experimental.rapidsmpf.frontend.ray import RayEngine

        # Prevent Ray from overriding ``CUDA_VISIBLE_DEVICES`` to ``""``
        # when a worker process starts with zero visible GPUs (e.g. the
        # test driver itself). In a future Ray release this becomes the
        # default; setting it eagerly here keeps test behaviour stable
        # across Ray versions.
        os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")
        _cached_ray_engine = RayEngine(
            **common_kwargs,
            num_ranks=NUM_RANKS,
            ray_init_options={"include_dashboard": False},
        )
    else:
        _cached_ray_engine._reset(**common_kwargs)  # type: ignore[attr-defined]
    return _cached_ray_engine


def shutdown_streaming_engine_cache() -> None:
    """
    Release the cached :class:`RayEngine`, if any.

    Test session-teardown fixtures **must** call this before
    ``ray.shutdown()`` so the engine's actor handles are released
    against a still-live cluster.
    """
    global _cached_ray_engine  # noqa: PLW0603
    if _cached_ray_engine is not None:  # pragma: no cover
        _cached_ray_engine.shutdown()
        _cached_ray_engine = None


def get_blocksize_mode(obj: pl.GPUEngine) -> Literal["medium", "small"]:
    """
    Infer the block size mode for a GPU engine.

    Parameters
    ----------
    obj
        Engine instance to inspect.

    Returns
    -------
      - "small": Engine is configured with a reduced ``max_rows_per_partition``.
      - "medium": Standard configuration, or non-streaming engine.
    """
    if not is_streaming_engine(obj):
        return "medium"

    # ``max_rows_per_partition`` is the defining signal for "small", it is
    # what triggers multi-partition behavior. Other fields in the small
    # baseline (``fallback_mode``, ``target_partition_size`` etc.) are
    # deliberately ignored so callers that override them on top of the
    # small baseline are still classified as ``"small"``.
    executor_options = obj.config["executor_options"]
    small_max_rows = create_streaming_options("small").max_rows_per_partition
    if executor_options.get("max_rows_per_partition") == small_max_rows:
        return "small"
    return "medium"
