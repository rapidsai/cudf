# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utilities for tracing and monitoring IR execution."""

from __future__ import annotations

import contextlib
import enum
import functools
import os
import threading
import time
import uuid
from typing import TYPE_CHECKING, Any, Concatenate, Literal, ParamSpec

import nvtx
import pynvml

import rmm
import rmm.statistics

from cudf_polars.utils.config import _bool_converter, get_device_handle

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

CUDF_POLARS_NVTX_DOMAIN = "cudf_polars"

nvtx_annotate_cudf_polars = functools.partial(
    nvtx.annotate, domain=CUDF_POLARS_NVTX_DOMAIN
)

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Sequence

    import cudf_polars.containers
    from cudf_polars.dsl import ir
    from cudf_polars.dsl.ir import IRExecutionContext
    from cudf_polars.quent import Task


def _dataframe_size_bytes(frame: cudf_polars.containers.DataFrame) -> int:
    return sum(col.device_buffer_size() for col in frame.table.columns())


def _begin_quent_do_evaluate_events(
    cls: type[ir.IR],
    ir_execution_context: IRExecutionContext,
) -> Task | None:
    import cudf_polars.quent

    quent_ir_execution_context = ir_execution_context.quent_ir_execution_context
    if quent_ir_execution_context is None:
        return None

    token = uuid.uuid4()
    quent_task = cudf_polars.quent.Task(
        instance_name=(
            f"{cls.__name__}-{quent_ir_execution_context.quent_operator.id.hex[:8]}-"
            f"{token.hex[:8]}"
        ),
        operator_id=quent_ir_execution_context.quent_operator.id,
    )
    quent_processor = quent_ir_execution_context.get_or_declare_processor(
        thread_ident=threading.get_ident(),
    )
    quent_ir_execution_context.logger.emit(quent_task.queueing())
    if not cls.is_io_node:
        quent_ir_execution_context.logger.emit(
            quent_task.allocating(resource_id=quent_processor.id)
        )
        quent_ir_execution_context.logger.emit(
            quent_task.computing(
                use_thread=quent_processor,
                use_memory=quent_ir_execution_context.device_memory,
                # memory_capacity_bytes=output_capacity_bytes,
            )
        )
    else:
        quent_ir_execution_context.logger.emit(
            quent_task.loading(
                use_thread=quent_processor,
                use_channel=quent_ir_execution_context.disk_to_device_channel,
                # channel_capacity_bytes=output_capacity_bytes,
                use_memory=quent_ir_execution_context.device_memory,
                # memory_capacity_bytes=output_capacity_bytes,
            )
        )

    return quent_task


def _end_quent_do_evaluate_events(
    cls: type[ir.IR],
    frames: Sequence[cudf_polars.containers.DataFrame],
    result: cudf_polars.containers.DataFrame,
    ir_execution_context: IRExecutionContext,
    quent_task: Task,
) -> None:
    import cudf_polars.quent

    quent_ir_execution_context = ir_execution_context.quent_ir_execution_context
    if quent_ir_execution_context is None:
        return

    output_capacity_bytes = _dataframe_size_bytes(result)
    quent_ir_execution_context.logger.emit(
        quent_ir_execution_context.quent_operator.statistics(
            statistics=cudf_polars.quent.Statistics(
                input_bytes=sum(_dataframe_size_bytes(frame) for frame in frames),
                output_bytes=output_capacity_bytes,
                output_rows=result.num_rows,
            )
        )
    )
    quent_ir_execution_context.logger.emit(quent_task.exit())


class Scope(enum.StrEnum):
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

    @functools.wraps(func)
    def wrapper(
        cls: type[ir.IR],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> cudf_polars.containers.DataFrame:
        ir_execution_context: IRExecutionContext | None = kwargs.get("context")  # type: ignore[assignment]

        frames: list[cudf_polars.containers.DataFrame] = (
            list(args) + [v for k, v in kwargs.items() if k != "context"]
        )[cls._n_non_child_args :]  # type: ignore[assignment]

        quent_task = None
        if ir_execution_context is not None:
            quent_task = _begin_quent_do_evaluate_events(cls, ir_execution_context)

        if LOG_TRACES:  # pragma: no cover; requires CUDF_POLARS_LOG_TRACES=1
            pynvml.nvmlInit()
            maybe_handle = get_device_handle()
            pid = _getpid()
            log = structlog.get_logger()

            before_start = time.monotonic_ns()
            before = make_snapshot(
                cls, frames, phase="input", device_handle=maybe_handle, pid=pid
            )
            before_end = time.monotonic_ns()
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
        else:
            result = func(cls, *args, **kwargs)

        if ir_execution_context is not None and quent_task is not None:
            _end_quent_do_evaluate_events(
                cls, frames, result, ir_execution_context, quent_task
            )

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
