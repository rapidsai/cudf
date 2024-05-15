# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Callback for the polars collect function to execute on device."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

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
    return ir.evaluate(cache={}).to_polars()


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
        callback = partial(_callback, translate_ir(nt))
    except NotImplementedError:
        return

    nt.set_udf(callback)
    return
