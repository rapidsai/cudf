# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Tracing."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import nvtx

if TYPE_CHECKING:
    import cudf_polars.containers.dataframe
    from cudf_polars.dsl.ir import IR


CUDF_POLARS_NVTX_DOMAIN = "cudf_polars"


def do_evaluate_with_tracing(
    cls: type[IR],
    *args: Any,
) -> cudf_polars.containers.dataframe.DataFrame:
    """
    Wrapper for IR.do_evaluate.

    This adds an nvtx annotation in the cudf_polars domain.

    Parameters
    ----------
    cls
        The type of the IR node to evaluate. Its ``do_evaluate``
        method will be called inside an nvtx annotation.
    args
        The arguments to pass to ``cls.do_evaluate``.

    Returns
    -------
    The result of the do_evaluate method.
    """
    with nvtx.annotate(
        message=cls.__name__,
        domain=CUDF_POLARS_NVTX_DOMAIN,
    ):
        return cls.do_evaluate(*args)
