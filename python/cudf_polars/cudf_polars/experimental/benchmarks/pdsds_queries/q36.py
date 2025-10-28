# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 36."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.utils import get_data

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 36."""
    return """
    SELECT Sum(ss_net_profit) / Sum(ss_ext_sales_price)                 AS
                   gross_margin,
                   i_category,
                   i_class,
                   Grouping(i_category) + Grouping(i_class)                     AS
                   lochierarchy,
                   Rank()
                     OVER (
                       partition BY Grouping(i_category)+Grouping(i_class), CASE
                     WHEN Grouping(
                     i_class) = 0 THEN i_category END
                       ORDER BY Sum(ss_net_profit)/Sum(ss_ext_sales_price) ASC) AS
                   rank_within_parent
    FROM   store_sales,
           date_dim d1,
           item,
           store
    WHERE  d1.d_year = 2000
           AND d1.d_date_sk = ss_sold_date_sk
           AND i_item_sk = ss_item_sk
           AND s_store_sk = ss_store_sk
           AND s_state IN ( 'TN', 'TN', 'TN', 'TN',
                            'TN', 'TN', 'TN', 'TN' )
    GROUP  BY rollup( i_category, i_class )
    ORDER  BY lochierarchy DESC,
              CASE
                WHEN lochierarchy = 0 THEN i_category
              END,
              rank_within_parent
    LIMIT 100;
    """


def level(  # noqa: D103
    base_data: pl.LazyFrame,
    group_cols: list[str],
    null_sentinel: str,
    lochierarchy: int,
) -> pl.LazyFrame:
    if group_cols:
        lf = base_data.group_by(group_cols).agg(
            [
                pl.col("ss_net_profit").sum().alias("total_net_profit"),
                pl.col("ss_ext_sales_price").sum().alias("total_ext_sales_price"),
            ]
        )
    else:
        lf = base_data.select(
            [
                pl.col("ss_net_profit").sum().alias("total_net_profit"),
                pl.col("ss_ext_sales_price").sum().alias("total_ext_sales_price"),
            ]
        )
    missing = [c for c in ["i_category", "i_class"] if c not in group_cols]
    if missing:
        lf = lf.with_columns([pl.lit(null_sentinel).alias(c) for c in missing])
    return lf.with_columns(
        [
            (pl.col("total_net_profit") / pl.col("total_ext_sales_price")).alias(
                "gross_margin"
            ),
            pl.lit(lochierarchy, dtype=pl.Int64).alias("lochierarchy"),
        ]
    ).select(["gross_margin", "i_category", "i_class", "lochierarchy"])


def polars_impl(run_config: RunConfig) -> pl.LazyFrame:
    """Query 36."""
    null_sentinel = "NULL"
    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    store = get_data(run_config.dataset_path, "store", run_config.suffix)
    item = get_data(run_config.dataset_path, "item", run_config.suffix)

    base_data = (
        store_sales.join(date_dim, left_on="ss_sold_date_sk", right_on="d_date_sk")
        .join(store, left_on="ss_store_sk", right_on="s_store_sk")
        .join(item, left_on="ss_item_sk", right_on="i_item_sk")
        .filter((pl.col("d_year") == 2000) & (pl.col("s_state").is_in(["TN"])))
    )

    level0 = level(base_data, ["i_category", "i_class"], null_sentinel, 0)
    level1 = level(base_data, ["i_category"], null_sentinel, 1)
    level2 = level(base_data, [], null_sentinel, 2)

    combined = pl.concat([level0, level1, level2])

    return (
        combined.with_columns(
            [
                pl.when(pl.col("lochierarchy") == 0)
                .then(pl.concat_str([pl.lit("0_"), pl.col("i_category")], separator=""))
                .when(pl.col("lochierarchy") == 1)
                .then(pl.lit("1"))
                .when(pl.col("lochierarchy") == 2)
                .then(pl.lit("2"))
                .alias("partition_key")
            ]
        )
        .with_columns(
            [
                pl.col("gross_margin")
                .rank("ordinal")
                .over("partition_key")
                .cast(pl.Int64)
                .alias("rank_within_parent")
            ]
        )
        .select(
            [
                "gross_margin",
                "i_category",
                "i_class",
                "lochierarchy",
                "rank_within_parent",
            ]
        )
        .sort(
            [
                pl.col("lochierarchy"),
                pl.when(pl.col("lochierarchy") == 0)
                .then(pl.col("i_category"))
                .otherwise(pl.lit(null_sentinel)),
                pl.col("rank_within_parent"),
            ],
            descending=[True, False, False],
            nulls_last=True,
        )
        .limit(100)
        .with_columns(
            [
                pl.when(pl.col("i_category") == null_sentinel)
                .then(None)
                .otherwise(pl.col("i_category"))
                .alias("i_category"),
                pl.when(pl.col("i_class") == null_sentinel)
                .then(None)
                .otherwise(pl.col("i_class"))
                .alias("i_class"),
            ]
        )
    )
