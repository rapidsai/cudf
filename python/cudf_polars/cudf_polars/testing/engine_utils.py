# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for :class:`polars.GPUEngine` tests."""

from __future__ import annotations

import importlib.util
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
        Block size mode, either ``"default"`` or ``"small"``.
    """

    full_name: str
    engine_name: str
    blocksize_mode: Literal["default", "small"]

    def __init__(self, full_name: str):
        self.full_name = full_name
        if full_name.endswith("-small"):
            self.engine_name = full_name[: -len("-small")]
            self.blocksize_mode = "small"
        else:
            # Covers ``"in-memory"`` and bare backend names like ``"spmd"``.
            self.engine_name = full_name
            self.blocksize_mode = "default"


def is_streaming_engine(obj: Any) -> bool:
    """Return ``True`` if ``obj`` is a :class:`StreamingEngine`."""
    try:
        from cudf_polars.experimental.rapidsmpf.frontend.core import StreamingEngine
    except ImportError:
        return False
    return isinstance(obj, StreamingEngine)


def get_blocksize_mode(obj: pl.GPUEngine) -> Literal["default", "small"]:
    """
    Infer the block size mode used to configure an engine.

    Parameters
    ----------
    obj
        Engine instance to inspect.

    Returns
    -------
    ``"small"`` if the engine's executor options match the small-partition
    configuration (``max_rows_per_partition == 4``), otherwise ``"default"``.
    Non-streaming engines always return ``"default"``.
    """
    if not is_streaming_engine(obj):
        return "default"
    executor_options = obj.config.get("executor_options", {})
    return "small" if executor_options.get("max_rows_per_partition") == 4 else "default"


def create_streaming_options(
    blocksize_mode: Literal["default", "small"],
    overrides: StreamingOptions | None = None,
) -> StreamingOptions:
    """
    Create :class:`StreamingOptions` for a block size mode.

    Parameters
    ----------
    blocksize_mode
        Block size configuration. ``"default"`` uses moderate partition sizes,
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
        case "default":
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
        case _:
            raise ValueError(f"Unknown blocksize_mode: {blocksize_mode!r}")
    if overrides is None:
        return baseline
    return StreamingOptions(**{**baseline.to_dict(), **overrides.to_dict()})


def build_streaming_engine(
    param: EngineFixtureParam,
    spmd_comm: Communicator,
    options: StreamingOptions | None = None,
) -> StreamingEngine:
    """
    Build a :class:`StreamingEngine` from an engine fixture parameter.

    Parameters
    ----------
    param
        Decoded engine fixture parameter describing the backend and block size mode.
    spmd_comm
        Communicator used when constructing an :class:`SPMDEngine`.
    options
        Optional streaming options to merge on top of the baseline selected by
        ``param.blocksize_mode``.

    Returns
    -------
    A streaming engine matching ``param``.
    """
    from cudf_polars.experimental.rapidsmpf.frontend.spmd import SPMDEngine

    streaming_options = create_streaming_options(param.blocksize_mode, options)
    match param.engine_name:
        case "spmd":
            return SPMDEngine(
                comm=spmd_comm,
                rapidsmpf_options=streaming_options.to_rapidsmpf_options(),
                executor_options=streaming_options.to_executor_options(),
                engine_options=streaming_options.to_engine_options(),
            )
        case _:
            raise AssertionError(f"Unknown streaming backend: {param.engine_name!r}")
