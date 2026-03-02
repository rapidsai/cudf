# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Utilities for tracing and monitoring IR execution."""

from __future__ import annotations

import enum
import functools
import os
import time
from typing import TYPE_CHECKING, Any, Concatenate, Literal

import nvtx
import pynvml
from typing_extensions import ParamSpec

import rmm
import rmm.statistics

from cudf_polars.utils.config import _bool_converter, get_device_handle

try:
    import structlog
except ImportError:
    _HAS_STRUCTLOG = False
else:
    _HAS_STRUCTLOG = True


LOG_TRACES = _HAS_STRUCTLOG and _bool_converter(
    os.environ.get("CUDF_POLARS_LOG_TRACES", "0")
)
LOG_MEMORY = LOG_TRACES and _bool_converter(
    os.environ.get("CUDF_POLARS_LOG_TRACES_MEMORY", "0")
)
LOG_DATAFRAMES = LOG_TRACES and _bool_converter(
    os.environ.get("CUDF_POLARS_LOG_TRACES_DATAFRAMES", "1")
)
LOG_COMPLETION_MEMORY = LOG_TRACES and _bool_converter(
    os.environ.get("CUDF_POLARS_LOG_TRACES_COMPLETION_MEMORY", "1")
)

CUDF_POLARS_NVTX_DOMAIN = "cudf_polars"

nvtx_annotate_cudf_polars = functools.partial(
    nvtx.annotate, domain=CUDF_POLARS_NVTX_DOMAIN
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import cudf_polars.containers
    from cudf_polars.dsl import ir


class Scope(str, enum.Enum):
    """Scope values for structured logging."""

    PLAN = "plan"
    ACTOR = "actor"
    EVALUATE_IR_NODE = "evaluate_ir_node"


@functools.cache
def _getpid() -> int:  # pragma: no cover
    # Gets called for each IR.do_evaluate node, so we'll cache it.
    return os.getpid()


def make_snapshot(
    node_type: type[ir.IR],
    frames: Sequence[cudf_polars.containers.DataFrame],
    extra: dict[str, Any] | None = None,
    *,
    pid: int,
    device_handle: Any | None = None,
    phase: Literal["input", "output"] = "input",
) -> dict:  # pragma: no cover; requires CUDF_POLARS_LOG_TRACES=1
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

    d: dict[str, Any] = {
        "type": ir_name,
    }

    if LOG_DATAFRAMES:
        d.update(
            {
                f"count_frames_{phase}": len(frames),
                f"frames_{phase}": [
                    {
                        "shape": frame.table.shape(),
                        "size": sum(
                            col.device_buffer_size() for col in frame.table.columns()
                        ),
                    }
                    for frame in frames
                ],
            }
        )
        d[f"total_bytes_{phase}"] = sum(x["size"] for x in d[f"frames_{phase}"])

    if LOG_MEMORY:
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

        if device_handle is not None:
            processes = pynvml.nvmlDeviceGetComputeRunningProcesses(device_handle)
            for proc in processes:
                if proc.pid == pid:
                    d[f"nvml_current_bytes_{phase}"] = proc.usedGpuMemory
                    break
    if extra:
        d.update(extra)

    return d


P = ParamSpec("P")


def log_do_evaluate(
    func: Callable[Concatenate[type[ir.IR], P], cudf_polars.containers.DataFrame],
) -> Callable[Concatenate[type[ir.IR], P], cudf_polars.containers.DataFrame]:
    """
    Decorator for an ``IR.do_evaluate`` method that logs information before and after evaluation.

    Parameters
    ----------
    func
        The ``IR.do_evaluate`` method to wrap.
    """
    if not LOG_TRACES:
        return func
    else:  # pragma: no cover; requires CUDF_POLARS_LOG_TRACES=1

        @functools.wraps(func)
        def wrapper(
            cls: type[ir.IR],
            *args: P.args,
            **kwargs: P.kwargs,
        ) -> cudf_polars.containers.DataFrame:
            # do this just once
            pynvml.nvmlInit()
            maybe_handle = get_device_handle()
            pid = _getpid()
            log = structlog.get_logger()

            # By convention, all non-dataframe arguments (non-child) come first.
            # Anything remaining is a dataframe, except for 'context' kwarg.
            frames: list[cudf_polars.containers.DataFrame] = (
                list(args) + [v for k, v in kwargs.items() if k != "context"]
            )[cls._n_non_child_args :]  # type: ignore[assignment]

            before_start = time.monotonic_ns()
            before = make_snapshot(
                cls, frames, phase="input", device_handle=maybe_handle, pid=pid
            )
            before_end = time.monotonic_ns()

            # The decorator preserves the exact signature of the original do_evaluate method.
            # Each IR.do_evaluate method is a classmethod that takes the IR class as first
            # argument, followed by the method-specific arguments, and returns a DataFrame.

            start = time.monotonic_ns()
            result = func(cls, *args, **kwargs)
            stop = time.monotonic_ns()

            after_start = time.monotonic_ns()
            after = make_snapshot(
                cls,
                [result],
                phase="output",
                extra={"start": start, "stop": stop},
                device_handle=maybe_handle,
                pid=pid,
            )
            after_end = time.monotonic_ns()
            record = (
                before
                | after
                | {
                    "scope": Scope.EVALUATE_IR_NODE.value,
                    "overhead_duration": (before_end - before_start)
                    + (after_end - after_start),
                }
            )
            log.info("Execute IR", **record)

            return result

        return wrapper


def log_pipeline_complete() -> None:
    """
    Log information about the pipeline completion.

    Notes
    -----
    LOG_TRACES must be enabled. This logs at the ``Scope.PLAN`` level.
    If RMM statistics are enabled, then the current allocation statistics are logged.
    """
    if not LOG_TRACES:
        return
    else:  # pragma: no cover; requires CUDF_POLARS_LOG_TRACES=1
        mr = rmm.mr.get_current_device_resource()
        if isinstance(mr, rmm.mr.StatisticsResourceAdaptor):
            stats = mr.allocation_counts
        elif isinstance(mr, rmm.mr.UpstreamResourceAdaptor) and isinstance(
            upstream_mr := mr.get_upstream(), rmm.mr.StatisticsResourceAdaptor
        ):
            stats = upstream_mr.allocation_counts
        else:
            stats = None

        log = structlog.get_logger()
        kwargs: dict[str, Any] = {}
        if stats:
            kwargs.update(
                rmm_current_bytes=stats.current_bytes,
                rmm_current_count=stats.current_count,
                rmm_peak_bytes=stats.peak_bytes,
                rmm_peak_count=stats.peak_count,
                rmm_total_bytes=stats.total_bytes,
                rmm_total_count=stats.total_count,
            )
        log.info(
            "Pipeline complete",
            **kwargs,
            scope=Scope.PLAN.value,
        )
