# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Tracing."""

from __future__ import annotations

import os
import time
import typing

import nvtx

import cudf_polars.containers.dataframe
import cudf_polars.utils.config

if typing.TYPE_CHECKING:
    from collections.abc import Callable


NVTX_ENABLED = os.environ.get("CUDF_POLARS_TRACE_NVTX", "").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
STRUCTLOG_ENABLED = os.environ.get("CUDF_POLARS_TRACE_STRUCTLOG", "").lower() in {
    "1",
    "true",
    "yes",
    "on",
}

if STRUCTLOG_ENABLED:
    try:
        import structlog
    except ImportError:
        message = (
            "'CUDF_POLARS_TRACE_STRUCTLOG' is set, but structlog is not installed."
        )
        raise ImportError(message) from None
    else:
        logger = structlog.get_logger()


class TracingOptions(typing.TypedDict):
    """Additional options provided to :func:`wrap_do_evaluate`."""

    name: str
    logger_options: dict[str, typing.Any]


def wrap_do_evaluate(
    ir_do_evaluate: Callable[..., cudf_polars.containers.dataframe.DataFrame],
    # args: tuple[typing.Any, ...],
    *args: typing.Any,
    wrapper_options: TracingOptions,
) -> cudf_polars.containers.dataframe.DataFrame:
    """
    Wrapper for IR.do_evaluate.

    Note that this appears in the Task Graph. We'd like to avoid placing the actual
    concrete IR nodes in the task graph, and so don't do things like that.
    """
    # *args2, wrapper_options = args

    if STRUCTLOG_ENABLED:
        payload: dict[str, typing.Any] = dict(wrapper_options["logger_options"])
        t0 = time.monotonic()

    if NVTX_ENABLED:
        rng = nvtx.start_range(
            message=wrapper_options["name"],
            domain="cudf_polars",
        )

    result = ir_do_evaluate(*args)

    if NVTX_ENABLED:
        nvtx.end_range(rng)

    if STRUCTLOG_ENABLED:
        if isinstance(result, cudf_polars.containers.dataframe.DataFrame):
            payload["output_dataframe"] = {
                "count_rows": result.num_rows,
                "count_columns": result.num_columns,
            }

        t1 = time.monotonic()
        logger.info(event="do_evaluate", **payload, duration=t1 - t0)

    return result
