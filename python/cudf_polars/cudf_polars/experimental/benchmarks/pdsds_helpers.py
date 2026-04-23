"""Shared helper functions for TPC-DS naive polars implementations."""
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Mapping


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
    """Build one rollup level: group, aggregate, fill missing cols, reorder."""
    if group_cols:
        lf = base.group_by(group_cols).agg(agg_exprs)
    else:
        lf = base.select(agg_exprs)

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

    if grouping_col is not None:
        lf = lf.with_columns(pl.lit(grouping_value, dtype=pl.Int64).alias(grouping_col))

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
    """Aggregate a sales channel with optional entity, returns, and extra joins."""
    lf = sales.join(dates, left_on=sales_date_key, right_on="d_date_sk")

    if entity_table is not None:
        lf = lf.join(entity_table, left_on=entity_key_sales, right_on=entity_key_dim)

    if extra_joins:
        for join_table, left_on, right_on in extra_joins:
            lf = lf.join(join_table, left_on=left_on, right_on=right_on)

    if returns_table is not None:
        returns_lf = returns_table
        if returns_date_key is not None:
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

    lf = lf.filter(date_filter)
    if extra_filters:
        for expr in extra_filters:
            lf = lf.filter(expr)

    if group_by_cols:
        lf = lf.group_by(group_by_cols).agg(agg_exprs)
    else:
        lf = lf.select(agg_exprs)

    return lf


def year_sales_agg(
    sales: pl.LazyFrame,
    date_dim: pl.LazyFrame,
    customer: pl.LazyFrame,
    *,
    sold_date_key: str,
    customer_key: str,
    customer_cols: list[str],
    agg_expr: pl.Expr,
    year_filter: pl.Expr | None = None,
) -> pl.LazyFrame:
    """Join sales to customer and date_dim, optionally filter by year, and aggregate."""
    lf = sales.join(customer, left_on=customer_key, right_on=customer_key)
    lf = lf.join(date_dim, left_on=sold_date_key, right_on="d_date_sk")
    if year_filter is not None:
        lf = lf.filter(year_filter)
    return lf.group_by([*customer_cols, "d_year"]).agg(agg_expr)
