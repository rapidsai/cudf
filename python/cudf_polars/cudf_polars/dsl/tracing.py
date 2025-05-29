# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Tracing."""

from __future__ import annotations

import os
import typing

import nvtx

if typing.TYPE_CHECKING:
    from collections.abc import Callable

    import cudf_polars.containers.dataframe


NVTX_ENABLED = os.environ.get("CUDF_POLARS_TRACE_NVTX", "").lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def wrap_do_evaluate(
    ir_do_evaluate: Callable[..., cudf_polars.containers.dataframe.DataFrame],
    *args: typing.Any,
    name: str,
) -> cudf_polars.containers.dataframe.DataFrame:
    """
    Wrapper for IR.do_evaluate.

    Note that this appears in the Task Graph. We'd like to avoid placing the actual
    concrete IR nodes in the task graph, and so don't do things like that.
    """
    # *args2, wrapper_options = args

    if NVTX_ENABLED:
        rng = nvtx.start_range(
            message=name,
            domain="cudf_polars",
        )

    result = ir_do_evaluate(*args)

    if NVTX_ENABLED:
        nvtx.end_range(rng)

    return result
