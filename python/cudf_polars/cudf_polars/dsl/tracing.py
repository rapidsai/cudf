# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Utilities for tracing and monitoring IR execution."""

from __future__ import annotations

import functools
import os
import time
from typing import TYPE_CHECKING, Any, Literal

import nvtx
import pynvml
from typing_extensions import ParamSpec

import rmm
import rmm.statistics

import cudf_polars.containers
from cudf_polars.utils.config import get_device_handle

try:
    import structlog
except ImportError:
    _HAS_STRUCTLOG = False
else:
    _HAS_STRUCTLOG = True


LOG_TRACES = _HAS_STRUCTLOG and os.environ.get(
    "CUDF_POLARS_LOG_TRACES", "0"
).lower() in {
    "1",
    "true",
    "y",
    "yes",
}

CUDF_POLARS_NVTX_DOMAIN = "cudf_polars"

nvtx_annotate_cudf_polars = functools.partial(
    nvtx.annotate, domain=CUDF_POLARS_NVTX_DOMAIN
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from cudf_polars.dsl import ir


def make_snaphot(
    node_type: type[ir.IR],
    frames: Sequence[cudf_polars.containers.DataFrame],
    extra: dict[str, Any] | None = None,
    *,
    pid: int,
    device_handle: Any | None = None,
    phase: Literal["input", "output"] = "input",
) -> dict:
    """
    Collect statistics about the evaluation of an IR node.

    Parameters
    ----------
    node_type
        The type of the IR node.
    frames
        The list of DataFrames to capture information for. For ``phase="input"``,
        this is typically the dataframes passed to ``IR.do_evaluate``. For
        ``phase="output"``, this is typically the DataFrame returned from
        ``IR.do_evaluate``.
    extra
        Extra information to log.
    pid
        The ID of the current process. Used for NVML memory usage.
    device_handle
        The pynvml device handle. Used for NVML memory usage.
    phase
        The phase of the evaluation. Either "input" or "output".
    """
    ir_name = node_type.__name__

    d = {
        "type": ir_name,
        f"count_frames_{phase}": len(frames),
        f"frames_{phase}": [
            {
                "shape": frame.table.shape(),
                "size": sum(col.device_buffer_size() for col in frame.table.columns()),
            }
            for frame in frames
        ],
    }
    d[f"total_bytes_{phase}"] = sum(x["size"] for x in d[f"frames_{phase}"])  # type: ignore[attr-defined]

    stats = rmm.statistics.get_statistics()
    if stats:
        d.update(
            {
                f"rmm_current_bytes_{phase}": stats.current_bytes,
                f"rmm_current_count_{phase}": stats.current_count,
                f"rmm_peak_bytes_{phase}": stats.peak_bytes,
                f"rmm_peak_count_{phase}": stats.peak_count,
                f"rmm_total_bytes_{phase}": stats.total_bytes,
                f"rmm_total_count_{phase}": stats.total_count,
            }
        )

    if extra:
        d.update(extra)

    if device_handle is not None:
        processes = pynvml.nvmlDeviceGetComputeRunningProcesses(device_handle)
        for proc in processes:
            if proc.pid == pid:
                d[f"nvml_current_bytes_{phase}"] = proc.usedGpuMemory
                break

    return d


P = ParamSpec("P")


def log_do_evaluate(
    func: Callable[P, cudf_polars.containers.DataFrame],
) -> Callable[P, cudf_polars.containers.DataFrame]:
    """
    Decorator for an ``IR.do_evaluate`` method that logs information before and after evaluation.

    Parameters
    ----------
    func
        The ``IR.do_evaluate`` method to wrap.
    """
    # do this just once
    pynvml.nvmlInit()
    maybe_handle = get_device_handle()

    pid = os.getpid()

    @functools.wraps(func)
    def wrapper(
        cls: type[ir.IR],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> cudf_polars.containers.DataFrame:
        if LOG_TRACES:
            log = structlog.get_logger()
            frames = [
                arg
                for arg in list(args) + list(kwargs.values())
                # TODO: See if this isinstance can be avoided.
                # Seems like `_non_child_args` gets us close...
                if isinstance(arg, cudf_polars.containers.DataFrame)
            ]

            before = make_snaphot(
                cls, frames, phase="input", device_handle=maybe_handle, pid=pid
            )

            # The types here aren't 100% correct.
            # We know that each IR.do_evaluate node is a
            # `Callable[[ir.Ir, <some other stuff>], cudf_polars.containers.DataFrame]`
            # but I'm not sure how to express that in a type annotation.

            start = time.monotonic_ns()
            result = func(cls, *args, **kwargs)  # type: ignore[arg-type]
            stop = time.monotonic_ns()

            after = make_snaphot(
                cls,
                [result],
                phase="output",
                extra={"start": start, "stop": stop},
                device_handle=maybe_handle,
                pid=pid,
            )
            record = before | after
            log.info("Execute IR", **record)

            return result
        else:
            return func(cls, *args, **kwargs)  # type: ignore[arg-type]

    return wrapper  # type: ignore[return-value]
