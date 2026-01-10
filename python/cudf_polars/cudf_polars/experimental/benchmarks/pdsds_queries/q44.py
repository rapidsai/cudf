# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 44."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.utils import get_data

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 44."""
    return """
    SELECT ascending.rnk,
                   i1.i_product_name best_performing,
                   i2.i_product_name worst_performing
    FROM  (SELECT *
           FROM   (SELECT item_sk,
                          Rank()
                            OVER (
                              ORDER BY rank_col ASC) rnk
                   FROM   (SELECT ss_item_sk         item_sk,
                                  Avg(ss_net_profit) rank_col
                           FROM   store_sales ss1
                           WHERE  ss_store_sk = 4
                           GROUP  BY ss_item_sk
                           HAVING Avg(ss_net_profit) > 0.9 *
                                  (SELECT Avg(ss_net_profit)
                                          rank_col
                                   FROM   store_sales
                                   WHERE  ss_store_sk = 4
                                          AND ss_cdemo_sk IS
                                              NULL
                                   GROUP  BY ss_store_sk))V1)
                  V11
           WHERE  rnk < 11) ascending,
          (SELECT *
           FROM   (SELECT item_sk,
                          Rank()
                            OVER (
                              ORDER BY rank_col DESC) rnk
                   FROM   (SELECT ss_item_sk         item_sk,
                                  Avg(ss_net_profit) rank_col
                           FROM   store_sales ss1
                           WHERE  ss_store_sk = 4
                           GROUP  BY ss_item_sk
                           HAVING Avg(ss_net_profit) > 0.9 *
                                  (SELECT Avg(ss_net_profit)
                                          rank_col
                                   FROM   store_sales
                                   WHERE  ss_store_sk = 4
                                          AND ss_cdemo_sk IS
                                              NULL
                                   GROUP  BY ss_store_sk))V2)
                  V21
           WHERE  rnk < 11) descending,
          item i1,
          item i2
    WHERE  ascending.rnk = descending.rnk
           AND i1.i_item_sk = ascending.item_sk
           AND i2.i_item_sk = descending.item_sk
    ORDER  BY ascending.rnk
    LIMIT 100;
    """


def polars_impl(run_config: RunConfig) -> pl.LazyFrame:
    """Query 44."""
    # Load tables
    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    item = get_data(run_config.dataset_path, "item", run_config.suffix)

    # Step 1: Calculate benchmark (average profit for store 4 with null demographics)
    benchmark = (
        store_sales.filter(
            (pl.col("ss_store_sk") == 4) & (pl.col("ss_cdemo_sk").is_null())
        )
        .group_by("ss_store_sk")
        .agg(
            [
                pl.col("ss_net_profit").mean().alias("profit_mean"),
                pl.col("ss_net_profit").count().alias("profit_count"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("profit_count") > 0)
                .then(pl.col("profit_mean"))
                .otherwise(None)
                .alias("benchmark_profit")
            ]
        )
        .select("benchmark_profit")
    )

    # Step 2: Calculate item-level average profits for store 4
    item_profits = (
        store_sales.filter(pl.col("ss_store_sk") == 4)
        .group_by("ss_item_sk")
        .agg(
            [
                pl.col("ss_net_profit").mean().alias("profit_mean"),
                pl.col("ss_net_profit").count().alias("profit_count"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("profit_count") > 0)
                .then(pl.col("profit_mean"))
                .otherwise(None)
                .alias("avg(ss_net_profit)")
            ]
        )
        .drop(["profit_mean", "profit_count"])
        .join(benchmark, how="cross")
        .filter(pl.col("avg(ss_net_profit)") > (0.9 * pl.col("benchmark_profit")))
    )

    # Step 3: Create ascending ranking (worst to best)
    ascending_rank = (
        item_profits.with_columns(
            [pl.col("avg(ss_net_profit)").rank(method="ordinal").alias("rnk")]
        )
        .filter(pl.col("rnk") < 11)
        .select(["ss_item_sk", "rnk"])
    )

    # Step 4: Create descending ranking (best to worst)
    descending_rank = (
        item_profits.with_columns(
            [
                pl.col("avg(ss_net_profit)")
                .rank(method="ordinal", descending=True)
                .alias("rnk")
            ]
        )
        .filter(pl.col("rnk") < 11)
        .select(["ss_item_sk", "rnk"])
    )

    # Step 5: Join rankings and get product names
    return (
        ascending_rank.join(descending_rank, on="rnk", how="inner", suffix="_desc")
        .join(item, left_on="ss_item_sk", right_on="i_item_sk", how="inner")
        .join(
            item,
            left_on="ss_item_sk_desc",
            right_on="i_item_sk",
            how="inner",
            suffix="_worst",
        )
        .select(
            [
                # Cast -> Int64 to match DuckDB
                pl.col("rnk").cast(pl.Int64),
                pl.col("i_product_name").alias("best_performing"),
                pl.col("i_product_name_worst").alias("worst_performing"),
            ]
        )
        .sort(["rnk"], nulls_last=True, descending=[False])
        .limit(100)
    )
