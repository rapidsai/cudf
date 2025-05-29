# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Tracing."""

from __future__ import annotations

import typing

import nvtx

if typing.TYPE_CHECKING:
    from collections.abc import Callable

    import cudf_polars.containers.dataframe


CUDF_POLARS_NVTX_DOMAIN = "cudf_polars"


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
    with nvtx.annotate(
        message=name,
        domain=CUDF_POLARS_NVTX_DOMAIN,
    ):
        return ir_do_evaluate(*args)
