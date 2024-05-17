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
    try:
        with nvtx.annotate(message="ExecuteIR", domain="cudf_polars"):
            return ir.evaluate(cache={}).to_polars()
    except Exception as e:
        print("Unable to evaluate", e)
        raise


def execute_with_cudf(nt) -> None:
    """
    A post optimization callback that attempts to execute the plan with cudf.

    Parameters
    ----------
    nt
        NodeTraverser

    The NodeTraverser is mutated if the libcudf executor can handle the plan.
    """
    try:
        with nvtx.annotate(message="ConvertIR", domain="cudf_polars"):
            callback = partial(_callback, translate_ir(nt))
    except NotImplementedError as e:
        print("Unable to translate", e)
        return

    nt.set_udf(callback)
    return
