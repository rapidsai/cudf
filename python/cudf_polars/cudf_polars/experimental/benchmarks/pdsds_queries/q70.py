# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 70."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.utils import get_data

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig
    from typing import Iterable


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 70."""
    return """
    SELECT Sum(ss_net_profit)                     AS total_sum, 
                   s_state, 
                   s_county, 
                   Grouping(s_state) + Grouping(s_county) AS lochierarchy, 
                   Rank() 
                     OVER ( 
                       partition BY Grouping(s_state)+Grouping(s_county), CASE WHEN 
                     Grouping( 
                     s_county) = 0 THEN s_state END 
                       ORDER BY Sum(ss_net_profit) DESC)  AS rank_within_parent 
    FROM   store_sales, 
           date_dim d1, 
           store 
    WHERE  d1.d_month_seq BETWEEN 1200 AND 1200 + 11 
           AND d1.d_date_sk = ss_sold_date_sk 
           AND s_store_sk = ss_store_sk 
           AND s_state IN (SELECT s_state 
                           FROM   (SELECT s_state                               AS 
                                          s_state, 
                                          Rank() 
                                            OVER ( 
                                              partition BY s_state 
                                              ORDER BY Sum(ss_net_profit) DESC) AS 
                                          ranking 
                                   FROM   store_sales, 
                                          store, 
                                          date_dim 
                                   WHERE  d_month_seq BETWEEN 1200 AND 1200 + 11 
                                          AND d_date_sk = ss_sold_date_sk 
                                          AND s_store_sk = ss_store_sk 
                                   GROUP  BY s_state) tmp1 
                           WHERE  ranking <= 5) 
    GROUP  BY rollup( s_state, s_county ) 
    ORDER  BY lochierarchy DESC, 
              CASE 
                WHEN lochierarchy = 0 THEN s_state 
              END, 
              rank_within_parent
    LIMIT 100;
    """


def _rollup_level(
    base: pl.LazyFrame,
    group_cols: Iterable[str],
    lochierarchy: int,
    null_state: bool = False,
    null_county: bool = False,
) -> pl.LazyFrame:
    group_cols = list(group_cols)
    if group_cols:
        out = base.group_by(group_cols).agg(pl.col("ss_net_profit").sum().alias("total_sum"))
    else:
        out = base.select(pl.col("ss_net_profit").sum().alias("total_sum"))

    s_state_expr = (
        pl.lit(None).cast(pl.String) if null_state or "s_state" not in group_cols else pl.col("s_state")
    )
    s_county_expr = (
        pl.lit(None).cast(pl.String) if null_county or "s_county" not in group_cols else pl.col("s_county")
    )

    return out.select(
        [
            pl.col("total_sum"),
            s_state_expr.alias("s_state"),
            s_county_expr.alias("s_county"),
            pl.lit(lochierarchy, dtype=pl.Int64).alias("lochierarchy"),
        ]
    )


def polars_impl(run_config: RunConfig) -> pl.LazyFrame:
    """Query 70."""
    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    store = get_data(run_config.dataset_path, "store", run_config.suffix)

    base = (
        store_sales
        .join(date_dim.select("d_date_sk", "d_month_seq"), left_on="ss_sold_date_sk", right_on="d_date_sk")
        .filter(pl.col("d_month_seq").is_between(1200, 1200 + 11))
        .join(store.select("s_store_sk", "s_state", "s_county"), left_on="ss_store_sk", right_on="s_store_sk")
    )

    lvl0 = _rollup_level(base, ["s_state", "s_county"], lochierarchy=0)
    lvl1 = _rollup_level(base, ["s_state"], lochierarchy=1, null_county=True)
    lvl2 = _rollup_level(base, [], lochierarchy=2, null_state=True, null_county=True)

    combined = pl.concat([lvl0, lvl1, lvl2])

    partition_key = (
        pl.when(pl.col("lochierarchy") == 0)
        .then(pl.concat_str([pl.lit("0|"), pl.col("s_state")], separator=""))
        .when(pl.col("lochierarchy") == 1)
        .then(pl.lit("1"))
        .otherwise(pl.lit("2"))
    )

    ranked = combined.with_columns(
        pl.col("total_sum").rank(method="dense", descending=True).over(partition_key).cast(pl.Int64).alias("rank_within_parent")
    )

    return (
        ranked
        .select(["total_sum", "s_state", "s_county", "lochierarchy", "rank_within_parent"])
        .sort(
            [
                pl.col("lochierarchy"),
                pl.when(pl.col("lochierarchy") == 0).then(pl.col("s_state")).otherwise(None),
                pl.col("rank_within_parent"),
            ],
            descending=[True, False, False],
            nulls_last=True,
        )
        .limit(100)
    )
