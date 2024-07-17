# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Callback for the polars collect function to execute on device."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import nvtx

from cudf_polars.dsl.translate import translate_ir

if TYPE_CHECKING:
    import polars as pl

    from cudf_polars.dsl.ir import IR
    from cudf_polars.typing import NodeTraverser

__all__: list[str] = ["execute_with_cudf"]


def _callback(
    ir: IR,
    with_columns: list[str] | None,
    pyarrow_predicate: str | None,
    n_rows: int | None,
) -> pl.DataFrame:
    assert with_columns is None
    assert pyarrow_predicate is None
    assert n_rows is None
    with nvtx.annotate(message="ExecuteIR", domain="cudf_polars"):
        return ir.evaluate(cache={}).to_polars()


def execute_with_cudf(
    nt: NodeTraverser,
    *,
    raise_on_fail: bool = False,
    exception: type[Exception] | tuple[type[Exception], ...] = Exception,
) -> None:
    """
    A post optimization callback that attempts to execute the plan with cudf.

    Parameters
    ----------
    nt
        NodeTraverser

    raise_on_fail
        Should conversion raise an exception rather than continuing
        without setting a callback.

    exception
        Optional exception, or tuple of exceptions, to catch during
        translation. Defaults to ``Exception``.

    The NodeTraverser is mutated if the libcudf executor can handle the plan.
    """
    try:
        with nvtx.annotate(message="ConvertIR", domain="cudf_polars"):
            nt.set_udf(partial(_callback, translate_ir(nt)))
    except exception:
        if raise_on_fail:
            raise
