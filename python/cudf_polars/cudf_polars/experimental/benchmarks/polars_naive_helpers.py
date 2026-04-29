"""Shared helper functions for TPC-DS naive polars implementations."""
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Generator

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Mapping


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


def rollup_level(
    base: pl.LazyFrame,
    group_cols: list[str],
    all_cols: Mapping[str, pl.DataType | type[pl.DataType]],
    agg_exprs: list[pl.Expr],
    output_order: list[str],
    *,
    grouping_col: str | None = None,
    grouping_value: int = 0,
    null_sentinel: str | None = None,
) -> pl.LazyFrame:
    """
    Build one rollup level: group, aggregate, fill missing cols, reorder.

    Implements one level of SQL's GROUP BY ROLLUP(...). For example, given
    ``GROUP BY ROLLUP(i_product_name, i_brand, i_class, i_category)``
    the SQL engine produces N+1 levels of aggregation (from most-detailed
    to grand total). This helper builds one of those levels.
    """
    # SQL: GROUP BY <group_cols> ... <agg_exprs>
    # When group_cols is non-empty, this produces one rollup level with
    # those columns retained. When empty, it computes the grand total.
    if group_cols:
        lf = base.group_by(group_cols).agg(agg_exprs)
    else:
        lf = base.select(agg_exprs)

    # SQL: Columns not in this rollup level's GROUP BY become NULL
    # (e.g., for level with only i_product_name, i_brand, i_class, i_category
    # is filled with NULL)
    missing = [col for col in all_cols if col not in group_cols]
    if missing:
        if null_sentinel is None:
            fill_exprs = [
                pl.lit(None, dtype=all_cols[col]).alias(col) for col in missing
            ]
        else:
            fill_exprs = [
                pl.lit(null_sentinel, dtype=all_cols[col]).alias(col) for col in missing
            ]
        lf = lf.with_columns(fill_exprs)

    # SQL: GROUPING() function — produces a 0/1 indicator for each rollup level
    if grouping_col is not None:
        lf = lf.with_columns(pl.lit(grouping_value, dtype=pl.Int64).alias(grouping_col))

    # SQL: SELECT <output_order> — reorder columns to match the SQL output
    return lf.select(output_order)


def channel_agg(
    sales: pl.LazyFrame,
    dates: pl.LazyFrame,
    *,
    sales_date_key: str,
    date_filter: pl.Expr,
    entity_table: pl.LazyFrame | None = None,
    entity_key_sales: str | None = None,
    entity_key_dim: str | None = None,
    returns_table: pl.LazyFrame | None = None,
    returns_date_key: str | None = None,
    returns_entity_key: str | None = None,
    agg_exprs: list[pl.Expr],
    group_by_cols: list[str],
    extra_joins: list[tuple[pl.LazyFrame, str, str]] | None = None,
    extra_filters: list[pl.Expr] | None = None,
) -> pl.LazyFrame:
    """
    Aggregate a sales channel with optional entity, returns, and extra joins.

    Implements the common per-channel CTE pattern found in TPC-DS queries
    like q5, q33, q56, q60, q77, q80::

        SELECT <group_by_cols>, <agg_exprs>
        FROM   <sales> JOIN <dates> ON <sales_date_key> = d_date_sk
               [JOIN <entity_table> ON <entity_key_sales> = <entity_key_dim>]
               [JOIN <extra_joins> ...]
               [LEFT JOIN <returns_table> ...]
        WHERE  <date_filter>
               [AND <extra_filters>]
        GROUP BY <group_by_cols>
    """
    # SQL: <sales> JOIN <dates> ON <sales_date_key> = d_date_sk
    lf = sales.join(dates, left_on=sales_date_key, right_on="d_date_sk")

    # SQL: JOIN <entity_table> ON <entity_key_sales> = <entity_key_dim>
    # (e.g., JOIN customer_address ON ss_addr_sk = ca_address_sk)
    if entity_table is not None:
        lf = lf.join(entity_table, left_on=entity_key_sales, right_on=entity_key_dim)

    # SQL: JOIN <extra tables> (e.g., JOIN item ON ss_item_sk = i_item_sk)
    if extra_joins:
        for join_table, left_on, right_on in extra_joins:
            lf = lf.join(join_table, left_on=left_on, right_on=right_on)

    # SQL: LEFT JOIN <returns_table> (e.g., LEFT JOIN store_returns)
    if returns_table is not None:
        returns_lf = returns_table
        if returns_date_key is not None:
            # SQL: returns JOIN <dates> ON <returns_date_key> = d_date_sk
            returns_lf = returns_lf.join(
                dates, left_on=returns_date_key, right_on="d_date_sk", suffix="_ret"
            )
        if returns_entity_key is not None:
            if entity_key_dim is not None or entity_key_sales is not None:
                lf = lf.join(
                    returns_lf,
                    left_on=entity_key_sales,
                    right_on=returns_entity_key,
                    how="left",
                )
            else:
                lf = lf.join(returns_lf, how="left")
        else:
            lf = lf.join(returns_lf, how="left")

    # SQL: WHERE <date_filter> [AND <extra_filters>]
    lf = lf.filter(date_filter)
    if extra_filters:
        for expr in extra_filters:
            lf = lf.filter(expr)

    # SQL: GROUP BY <group_by_cols> ... SELECT <agg_exprs>
    if group_by_cols:
        lf = lf.group_by(group_by_cols).agg(agg_exprs)
    else:
        lf = lf.select(agg_exprs)

    return lf
