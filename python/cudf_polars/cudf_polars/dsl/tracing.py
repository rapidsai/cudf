# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Tracing."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

import nvtx

if TYPE_CHECKING:
    from collections.abc import Callable

    import cudf_polars.containers.dataframe
    from cudf_polars.dsl.ir import IR


CUDF_POLARS_NVTX_DOMAIN = "cudf_polars"


def do_evaluate_with_tracing(
    ir: IR,
) -> Callable[..., cudf_polars.containers.dataframe.DataFrame]:
    """
    Wrapper for IR.do_evaluate.

    This adds an nvtx annotation in the cudf_polars domain.

    Parameters
    ----------
    ir
        The IR node to evaluate. Its ``do_evaluate`` method will be
        called inside an nvtx annotation.

    Returns
    -------
    The result of the do_evaluate method.
    """
    cls = type(ir)

    @functools.wraps(cls.do_evaluate)
    def wrapped(*args: Any) -> cudf_polars.containers.dataframe.DataFrame:
        with nvtx.annotate(
            message=cls.__name__,
            domain=CUDF_POLARS_NVTX_DOMAIN,
        ):
            return cls.do_evaluate(*args)

    return wrapped
