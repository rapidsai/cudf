# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""SQL-compatible aggregation helpers for TPC-DS benchmarks."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Generator

import polars as pl

if TYPE_CHECKING:
    pass


@contextmanager
def sql_semantics() -> Generator[None, None, None]:
    """Patch polars.Expr.sum to match SQL null semantics for the duration of the block.

    In standard SQL, SUM over an all-null group returns NULL.
    Polars returns 0.  Wrapping query-plan construction in this context manager
    corrects that discrepancy so results validate correctly against DuckDB.
    See https://github.com/rapidsai/cudf/issues/19560.
    """
    _original_sum = pl.Expr.sum

    def _sql_sum(self: pl.Expr) -> pl.Expr:
        return pl.when(self.count() > 0).then(_original_sum(self)).otherwise(None)

    pl.Expr.sum = _sql_sum  # type: ignore[method-assign]
    try:
        yield
    finally:
        pl.Expr.sum = _original_sum
