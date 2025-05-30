# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Tracing."""

from __future__ import annotations

import functools
import typing

import nvtx

if typing.TYPE_CHECKING:
    from collections.abc import Callable

    import cudf_polars.containers.dataframe


CUDF_POLARS_NVTX_DOMAIN = "cudf_polars"


def do_evaluate_traced(
    ir_do_evaluate: Callable[..., cudf_polars.containers.dataframe.DataFrame],
    *,
    name: str,
) -> Callable[..., cudf_polars.containers.dataframe.DataFrame]:
    """
    Wrapper for IR.do_evaluate.

    This adds an nvtx annotation in the cudf_polars domain.

    Parameters
    ----------
    ir_do_evaluate
        The do_evaluate method of an IR node.
    name
        The name of the IR node, typically from ``type(ir).__name__``.
        This will be used as the nvtx message.

    Returns
    -------
    The result of the do_evaluate method.
    """

    @functools.wraps(ir_do_evaluate)
    def wrapped(*args: typing.Any) -> cudf_polars.containers.dataframe.DataFrame:
        with nvtx.annotate(
            message=name,
            domain=CUDF_POLARS_NVTX_DOMAIN,
        ):
            return ir_do_evaluate(*args)

    return wrapped
