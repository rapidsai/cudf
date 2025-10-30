# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 22."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.utils import get_data

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 22."""
    return """
    SELECT i_product_name,
                   i_brand,
                   i_class,
                   i_category,
                   Avg(inv_quantity_on_hand) qoh
    FROM   inventory,
           date_dim,
           item,
           warehouse
    WHERE  inv_date_sk = d_date_sk
           AND inv_item_sk = i_item_sk
           AND inv_warehouse_sk = w_warehouse_sk
           AND d_month_seq BETWEEN 1205 AND 1205 + 11
    GROUP  BY rollup( i_product_name, i_brand, i_class, i_category )
    ORDER  BY qoh,
              i_product_name,
              i_brand,
              i_class,
              i_category
    LIMIT 100;
    """


def level(  # noqa: D103
    base_data: pl.LazyFrame, agg_exprs: list[pl.Expr], group_cols: list[str]
) -> pl.LazyFrame:
    if group_cols:
        lf = base_data.group_by(group_cols).agg(agg_exprs)
    else:
        lf = base_data.select(agg_exprs)
    missing = [
        c
        for c in ["i_product_name", "i_brand", "i_class", "i_category"]
        if c not in group_cols
    ]
    if missing:
        lf = lf.with_columns([pl.lit(None, dtype=pl.String).alias(c) for c in missing])
    return lf.select(
        [
            "i_product_name",
            "i_brand",
            "i_class",
            "i_category",
            "qoh",
        ]
    )


def polars_impl(run_config: RunConfig) -> pl.LazyFrame:
    """Query 22."""
    inventory = get_data(run_config.dataset_path, "inventory", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    item = get_data(run_config.dataset_path, "item", run_config.suffix)
    warehouse = get_data(run_config.dataset_path, "warehouse", run_config.suffix)
    base_data = (
        inventory.join(date_dim, left_on="inv_date_sk", right_on="d_date_sk")
        .join(item, left_on="inv_item_sk", right_on="i_item_sk")
        .join(warehouse, left_on="inv_warehouse_sk", right_on="w_warehouse_sk")
        .filter(pl.col("d_month_seq").is_between(1205, 1205 + 11))
    )
    agg_exprs = [pl.col("inv_quantity_on_hand").mean().alias("qoh")]

    level1 = level(
        base_data, agg_exprs, ["i_product_name", "i_brand", "i_class", "i_category"]
    )
    level2 = level(base_data, agg_exprs, ["i_product_name", "i_brand", "i_class"])
    level3 = level(base_data, agg_exprs, ["i_product_name", "i_brand"])
    level4 = level(base_data, agg_exprs, ["i_product_name"])
    level5 = level(base_data, agg_exprs, [])

    return (
        pl.concat([level1, level2, level3, level4, level5])
        .sort(
            ["qoh", "i_product_name", "i_brand", "i_class", "i_category"],
            nulls_last=True,
        )
        .limit(100)
    )
