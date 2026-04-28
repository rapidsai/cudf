# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 44."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.pdsds_parameters import load_parameters
from cudf_polars.experimental.benchmarks.utils import QueryResult, get_data

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 44."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=44,
        qualification=run_config.qualification,
    )

    store_sk = params["store_sk"]

    return f"""
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
                           WHERE  ss_store_sk = {store_sk}
                           GROUP  BY ss_item_sk
                           HAVING Avg(ss_net_profit) > 0.9 *
                                  (SELECT Avg(ss_net_profit)
                                          rank_col
                                   FROM   store_sales
                                   WHERE  ss_store_sk = {store_sk}
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
                           WHERE  ss_store_sk = {store_sk}
                           GROUP  BY ss_item_sk
                           HAVING Avg(ss_net_profit) > 0.9 *
                                  (SELECT Avg(ss_net_profit)
                                          rank_col
                                   FROM   store_sales
                                   WHERE  ss_store_sk = {store_sk}
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


def polars_impl(run_config: RunConfig) -> QueryResult:
    """Query 44."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=44,
        qualification=run_config.qualification,
    )

    store_sk = params["store_sk"]

    # Load tables
    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    item = get_data(run_config.dataset_path, "item", run_config.suffix)

    # Step 1: Calculate benchmark (average profit for store with null demographics)
    benchmark = (
        store_sales.filter(
            (pl.col("ss_store_sk") == store_sk) & (pl.col("ss_cdemo_sk").is_null())
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

    # Step 2: Calculate item-level average profits for store
    item_profits = (
        store_sales.filter(pl.col("ss_store_sk") == store_sk)
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

    sort_by = {"rnk": False}
    limit = 100
    # Step 5: Join rankings and get product names
    return QueryResult(
        frame=(
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
                    pl.col("rnk"),
                    pl.col("i_product_name").alias("best_performing"),
                    pl.col("i_product_name_worst").alias("worst_performing"),
                ]
            )
            .sort(sort_by.keys(), nulls_last=True)
            .limit(limit)
        ),
        sort_by=list(sort_by.items()),
        limit=limit,
    )


def polars_impl_naive(run_config: RunConfig) -> QueryResult:
    """Query 44 (naive)."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=44,
        qualification=run_config.qualification,
    )

    store_sk = params["store_sk"]

    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    item = get_data(run_config.dataset_path, "item", run_config.suffix)

    # SQL: benchmark — Avg(ss_net_profit) WHERE ss_store_sk = {store_sk} AND ss_cdemo_sk IS NULL
    benchmark = (
        # SQL: WHERE ss_store_sk = {store_sk} AND ss_cdemo_sk IS NULL
        store_sales.filter(
            (pl.col("ss_store_sk") == store_sk) & (pl.col("ss_cdemo_sk").is_null())
        )
        # SQL: GROUP BY ss_store_sk
        .group_by("ss_store_sk")
        .agg(pl.col("ss_net_profit").mean().alias("benchmark_profit"))
        .select("benchmark_profit")
        .with_columns(pl.lit(1).alias("join_key"))
    )

    # SQL: inner subquery shared by ascending and descending — item-level avg profit filtered
    # against the benchmark. Polars' comm_subplan_elim optimization detects that both branches
    # reference an identical subtree and inserts a CACHE node, so sharing a variable here does
    # not give an unfair advantage over inlining the definition into each branch.
    item_profits = (
        # SQL: WHERE ss_store_sk = {store_sk}
        store_sales.filter(pl.col("ss_store_sk") == store_sk)
        # SQL: GROUP BY ss_item_sk
        .group_by("ss_item_sk")
        # SQL: Avg(ss_net_profit) AS avg(ss_net_profit)
        .agg(pl.col("ss_net_profit").mean().alias("avg(ss_net_profit)"))
        .with_columns(pl.lit(1).alias("join_key"))
        .join(benchmark, on="join_key")
        .filter(pl.col("avg(ss_net_profit)") > (0.9 * pl.col("benchmark_profit")))
        .select(["ss_item_sk", "avg(ss_net_profit)"])
    )

    rank_limit = 11
    # SQL: ascending rank — Rank() OVER (ORDER BY rank_col ASC) WHERE rnk < 11
    ascending_rank = (
        item_profits.with_columns(
            [pl.col("avg(ss_net_profit)").rank(method="ordinal").alias("rnk")]
        )
        .filter(pl.col("rnk") < rank_limit)
        .select(["ss_item_sk", "rnk"])
    )
    # SQL: descending rank — Rank() OVER (ORDER BY rank_col DESC) WHERE rnk < 11
    descending_rank = (
        item_profits.with_columns(
            [
                pl.col("avg(ss_net_profit)")
                .rank(method="ordinal", descending=True)
                .alias("rnk")
            ]
        )
        .filter(pl.col("rnk") < rank_limit)
        .select(["ss_item_sk", "rnk"])
    )

    sort_by = {"rnk": False}
    limit = 100
    return QueryResult(
        frame=(
            # SQL: JOIN ascending ON rnk = rnk
            ascending_rank.join(descending_rank, on="rnk", how="inner", suffix="_desc")
            # SQL: JOIN item i1 ON ss_item_sk = i_item_sk (best_performing)
            .join(item, left_on="ss_item_sk", right_on="i_item_sk", how="inner")
            # SQL: JOIN item i2 ON ss_item_sk_desc = i_item_sk (worst_performing)
            .join(
                item,
                left_on="ss_item_sk_desc",
                right_on="i_item_sk",
                how="inner",
                suffix="_worst",
            )
            # SQL: SELECT rnk, i1.i_product_name AS best_performing, i2.i_product_name AS worst_performing
            .select(
                [
                    pl.col("rnk"),
                    pl.col("i_product_name").alias("best_performing"),
                    pl.col("i_product_name_worst").alias("worst_performing"),
                ]
            )
            # SQL: ORDER BY rnk
            .sort(sort_by.keys(), nulls_last=True)
            # SQL: LIMIT 100
            .limit(limit)
        ),
        sort_by=list(sort_by.items()),
        limit=limit,
    )
