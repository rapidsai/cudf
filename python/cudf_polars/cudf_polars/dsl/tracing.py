# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Utilities for tracing and monitoring IR execution."""

from __future__ import annotations

import contextlib
import enum
import functools
import os
import time
import uuid
from typing import TYPE_CHECKING, Any, Concatenate, Literal, ParamSpec

import nvtx
import pynvml

import rmm
import rmm.statistics

from cudf_polars.utils.config import _bool_converter, get_device_handle
from cudf_polars.utils.cuda_stream import get_joined_cuda_stream

try:  # pragma: no cover; requires structlog
    import structlog
except ImportError:  # pragma: no cover; requires no structlog
    _HAS_STRUCTLOG = False
else:  # pragma: no cover; requires structlog
    _HAS_STRUCTLOG = True


LOG_TRACES = _HAS_STRUCTLOG and _bool_converter(
    os.environ.get("CUDF_POLARS_LOG_TRACES", "0")
)
LOG_MEMORY = LOG_TRACES and _bool_converter(
    os.environ.get("CUDF_POLARS_LOG_TRACES_MEMORY", "1")
)
LOG_DATAFRAMES = LOG_TRACES and _bool_converter(
    os.environ.get("CUDF_POLARS_LOG_TRACES_DATAFRAMES", "1")
)
LOG_TRACES_GPU = LOG_TRACES and _bool_converter(
    os.environ.get("CUDF_POLARS_LOG_TRACES_GPU", "0")
)

CUDF_POLARS_NVTX_DOMAIN = "cudf_polars"

nvtx_annotate_cudf_polars = functools.partial(
    nvtx.annotate, domain=CUDF_POLARS_NVTX_DOMAIN
)

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Sequence

    import cudf_polars.containers
    from cudf_polars.dsl import ir


class Scope(str, enum.Enum):
    """Scope values for structured logging."""

    PLAN = "plan"
    ACTOR = "actor"
    EVALUATE_IR_NODE = "evaluate_ir_node"
    EVALUATE_IR_NODE_GPU = "evaluate_ir_node_gpu"


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
            from cudf_polars.dsl import tracing_gpu as _tracing_gpu

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

            trace_event_id: str | None = None
            query_id_str: str | None = None
            gpu_events = None
            if LOG_TRACES_GPU:
                # By convention, kwargs["context"] is an IRExecutionContext
                exec_ctx: ir.IRExecutionContext = kwargs["context"]  # type: ignore[assignment]

                # The GPU trace is emitted on a different Python thread at some point in
                # the future, so the context setting actor IDs, etc. might not be available.
                # We can link GPU traces to a host task through this UUID, and from there
                # to things like the actor ID.
                trace_event_id = uuid.uuid4().hex
                query_id_str = str(exec_ctx.query_id)
                timing_stream = get_joined_cuda_stream(
                    exec_ctx.get_cuda_stream,
                    upstreams=[frame.stream for frame in frames],
                )
                gpu_events = _tracing_gpu.begin_gpu_interval(timing_stream)

            start = time.monotonic_ns()
            try:
                result = func(cls, *args, **kwargs)
            except BaseException:
                if LOG_TRACES_GPU and gpu_events is not None:
                    ev_s, ev_e = gpu_events
                    _tracing_gpu.destroy_event_pair(ev_s, ev_e)
                raise
            stop = time.monotonic_ns()
            snapshot_extra: dict[str, Any] = {"start": start, "stop": stop}

            if LOG_TRACES_GPU:
                assert gpu_events is not None
                assert trace_event_id is not None
                assert query_id_str is not None

                ev_s, ev_e = gpu_events
                # Record end on ``result.stream`` after IR work, then join a dedicated
                # notify stream downstream before ``cudaLaunchHostFunc`` (see tracing_gpu).
                ok, gpu_err = _tracing_gpu.enqueue_gpu_trace_completion(
                    interval_end_stream=result.stream,
                    host_notify_stream=_tracing_gpu.get_host_notify_stream(),
                    ev_start=ev_s,
                    ev_end=ev_e,
                    trace_event_id=trace_event_id,
                    query_id=query_id_str,
                    ir_type=cls.__name__,
                    log=log,
                )
                if not ok:
                    log.warning(
                        "Execute IR GPU scheduling failed",
                        scope=Scope.EVALUATE_IR_NODE_GPU.value,
                        trace_event_id=trace_event_id,
                        query_id=query_id_str,
                        type=cls.__name__,
                        error=gpu_err,
                    )

                snapshot_extra["trace_event_id"] = trace_event_id
                snapshot_extra["query_id"] = query_id_str

            after_start = time.monotonic_ns()
            after = make_snapshot(
                cls,
                [result],
                phase="output",
                extra=snapshot_extra,
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


@contextlib.contextmanager
def bound_contextvars(**kwargs: Any) -> Generator[None, None, None]:
    """Wrapper around structlog.contextvars.bound_contextvars."""
    if LOG_TRACES:  # pragma: no cover; requires CUDF_POLARS_LOG_TRACES=1
        with structlog.contextvars.bound_contextvars(**kwargs):
            yield
    else:
        yield


def log(message: str, **kwargs: Any) -> None:
    """Wrapper around structlog.get_logger().info."""
    if LOG_TRACES:  # pragma: no cover; requires CUDF_POLARS_LOG_TRACES=1
        log = structlog.get_logger()
        log.info(message, **kwargs)
