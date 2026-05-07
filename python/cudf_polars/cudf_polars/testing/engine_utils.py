# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for :class:`polars.GPUEngine` tests."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Mapping
    from contextlib import AbstractContextManager

    import polars as pl

    from cudf_polars.experimental.rapidsmpf.frontend.core import StreamingEngine
    from cudf_polars.experimental.rapidsmpf.frontend.options import StreamingOptions


STREAMING_ENGINE_FIXTURE_PARAMS: list[str] = []
if importlib.util.find_spec("rapidsmpf") is not None:
    STREAMING_ENGINE_FIXTURE_PARAMS.extend(["spmd", "spmd-small"])
    # ``DaskEngine`` and ``RayEngine`` both reject construction inside an
    # ``rrun`` cluster.
    from rapidsmpf.bootstrap import is_running_with_rrun as _is_running_with_rrun

    if not _is_running_with_rrun():  # pragma: no cover
        if importlib.util.find_spec("distributed") is not None:
            STREAMING_ENGINE_FIXTURE_PARAMS.append("dask")
        if importlib.util.find_spec("ray") is not None:
            STREAMING_ENGINE_FIXTURE_PARAMS.append("ray")
ALL_ENGINE_FIXTURE_PARAMS = ["in-memory", *STREAMING_ENGINE_FIXTURE_PARAMS]


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


def warns_on_spmd(  # pragma: no cover; rapidsmpf-only path
    engine: Any,
    *args: Any,
    when: bool = True,
    **kwargs: Any,
) -> AbstractContextManager[Any]:
    """
    ``pytest.warns(*args, **kwargs)`` on SPMD; ``nullcontext`` otherwise.

    ``pytest.warns`` only captures warnings emitted in the test process. On
    multi-process backends (``DaskEngine``, ``RayEngine``) the fallback
    warning fires on workers/actors and only appears in worker logs/stdout,
    so the assertion is replaced with a passthrough on those backends.

    The optional ``when`` kwarg lets callers compose an additional gate (e.g.
    a parametrize value) without an outer ``if``.
    """
    import contextlib

    import pytest

    from cudf_polars.experimental.rapidsmpf.frontend.spmd import SPMDEngine

    if when and isinstance(engine, SPMDEngine):
        return pytest.warns(*args, **kwargs)
    return contextlib.nullcontext()


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

    # ``allow_gpu_sharing=True`` is always set so the cached multi-rank
    # engines (Dask workers, Ray actors with ``num_ranks > 1``) don't trip
    # the UUID-collision guard on every ``_reset(...)``.
    match blocksize_mode:
        case "medium":
            baseline = StreamingOptions(
                max_rows_per_partition=50,
                dynamic_planning={},
                target_partition_size=1_000_000,
                raise_on_fail=True,
                allow_gpu_sharing=True,
            )
        case "small":
            baseline = StreamingOptions(
                max_rows_per_partition=4,
                dynamic_planning={},
                target_partition_size=10,
                raise_on_fail=True,
                fallback_mode=StreamingFallbackMode.SILENT,
                allow_gpu_sharing=True,
            )
        case _:  # pragma: no cover
            raise ValueError(f"Unknown blocksize_mode: {blocksize_mode!r}")
    if overrides is None:
        return baseline
    return StreamingOptions(**{**baseline.to_dict(), **overrides.to_dict()})


def build_streaming_engine(
    param: EngineFixtureParam,
    engines: Mapping[str, StreamingEngine],
    options: StreamingOptions | None = None,
) -> StreamingEngine:
    """
    Return ``engines``'s entry for ``param``, ``_reset``-ed.

    ``engines`` must already contain a slot for ``param.engine_name`` —
    seeded by the ``streaming_engines`` session-scoped fixture. The
    fixture owns mutation; this function only reads and ``_reset``-s.

    Parameters
    ----------
    param
        Decoded engine fixture parameter describing the backend and block size mode.
    engines
        Streaming-engine collection keyed by backend name. Provided by
        the ``streaming_engines`` test fixture.
    options
        Optional streaming options to merge on top of the baseline selected by
        ``param.blocksize_mode``.

    Returns
    -------
    The shared :class:`StreamingEngine`, ``_reset`` to the requested options.

    Raises
    ------
    RuntimeError
        If ``engines`` has no slot for ``param.engine_name``.
    """
    streaming_options = create_streaming_options(param.blocksize_mode, options)
    engine = engines.get(param.engine_name)
    if engine is None:  # pragma: no cover
        raise RuntimeError(
            f"No streaming engine for {param.engine_name!r}. The corresponding "
            "session-scoped fixture must populate the collection before tests run."
        )
    engine._reset(
        rapidsmpf_options=streaming_options.to_rapidsmpf_options(),
        executor_options=streaming_options.to_executor_options(),
        engine_options=streaming_options.to_engine_options(),
    )
    return engine


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
