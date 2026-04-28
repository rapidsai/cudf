# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 70."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.pdsds_parameters import load_parameters
from cudf_polars.experimental.benchmarks.polars_naive_helpers import rollup_level
from cudf_polars.experimental.benchmarks.utils import QueryResult, get_data

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 70."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=70,
        qualification=run_config.qualification,
    )

    dms = params["dms"]

    return f"""
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
    WHERE  d1.d_month_seq BETWEEN {dms} AND {dms} + 11
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
                                   WHERE  d_month_seq BETWEEN {dms} AND {dms} + 11
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


def polars_impl(run_config: RunConfig) -> QueryResult:
    """Query 70."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=70,
        qualification=run_config.qualification,
    )

    dms = params["dms"]

    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    store = get_data(run_config.dataset_path, "store", run_config.suffix)

    top_states = (
        store_sales.join(date_dim, left_on="ss_sold_date_sk", right_on="d_date_sk")
        .join(store, left_on="ss_store_sk", right_on="s_store_sk")
        .filter(pl.col("d_month_seq").is_between(dms, dms + 11))
        .select("s_state")
        .unique()
    )

    base = (
        store_sales.join(date_dim, left_on="ss_sold_date_sk", right_on="d_date_sk")
        .join(store, left_on="ss_store_sk", right_on="s_store_sk")
        .filter(pl.col("d_month_seq").is_between(dms, dms + 11))
        .join(top_states, on="s_state", how="semi")
    )

    detail = (
        base.group_by(["s_state", "s_county"])
        .agg(pl.col("ss_net_profit").sum().alias("total_sum"))
        .with_columns(pl.lit(0, dtype=pl.Int64).alias("lochierarchy"))
    )

    by_state = (
        detail.group_by("s_state")
        .agg(pl.col("total_sum").sum())
        .with_columns(
            [
                pl.lit(None, dtype=pl.Utf8).alias("s_county"),
                pl.lit(1, dtype=pl.Int64).alias("lochierarchy"),
            ]
        )
        .select(["s_state", "s_county", "total_sum", "lochierarchy"])
    )

    total = (
        detail.select(pl.col("total_sum").sum())
        .with_columns(
            [
                pl.lit(None, dtype=pl.Utf8).alias("s_state"),
                pl.lit(None, dtype=pl.Utf8).alias("s_county"),
                pl.lit(2, dtype=pl.Int64).alias("lochierarchy"),
            ]
        )
        .select(["s_state", "s_county", "total_sum", "lochierarchy"])
    )

    return QueryResult(
        frame=(
            pl.concat([detail, by_state, total])
            .with_columns(
                pl.when(pl.col("lochierarchy") == 0)
                .then(pl.col("s_state"))
                .otherwise(None)
                .alias("partition_key")
            )
            .with_columns(
                pl.col("total_sum")
                .rank(method="min", descending=True)
                .over(["lochierarchy", "partition_key"])
                .alias("rank_within_parent")
            )
            .select(
                [
                    "total_sum",
                    "s_state",
                    "s_county",
                    "lochierarchy",
                    "rank_within_parent",
                ]
            )
            .sort(
                [
                    pl.col("lochierarchy"),
                    pl.when(pl.col("lochierarchy") == 0)
                    .then(pl.col("s_state"))
                    .otherwise(None),
                    pl.col("rank_within_parent"),
                ],
                descending=[True, False, False],
                nulls_last=True,
            )
            .limit(100)
        ),
        sort_by=[("lochierarchy", True), ("rank_within_parent", False)],
        limit=100,
        sort_keys=[
            (pl.col("lochierarchy"), True),
            (
                pl.when(pl.col("lochierarchy") == 0)
                .then(pl.col("s_state"))
                .otherwise(None),
                False,
            ),
            (pl.col("rank_within_parent"), False),
        ],
    )


def polars_impl_naive(run_config: RunConfig) -> QueryResult:
    """Query 70 (naive)."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=70,
        qualification=run_config.qualification,
    )

    dms = params["dms"]

    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    store = get_data(run_config.dataset_path, "store", run_config.suffix)

    # SQL: tmp1 — correlated subquery to find top-5 states by net profit
    # Note: RANK() OVER (PARTITION BY s_state) on a GROUP BY s_state result always
    # gives rank=1 (one row per partition), so ranking <= 5 keeps all states.
    # We translate the SQL literally here rather than short-circuiting.
    tmp1 = (
        store_sales.join(store, left_on="ss_store_sk", right_on="s_store_sk")
        .join(date_dim, left_on="ss_sold_date_sk", right_on="d_date_sk")
        .filter(pl.col("d_month_seq").is_between(dms, dms + 11))
        # SQL: GROUP BY s_state
        .group_by("s_state")
        .agg(pl.col("ss_net_profit").sum().alias("total_profit"))
        # SQL: Rank() OVER (PARTITION BY s_state ORDER BY Sum(ss_net_profit) DESC)
        .with_columns(
            pl.col("total_profit")
            .rank(method="min", descending=True)
            .over("s_state")
            .alias("ranking")
        )
        # SQL: WHERE ranking <= 5
        .filter(pl.col("ranking") <= 5)
        .select("s_state")
    )

    # SQL: CTE base — FROM store_sales, date_dim, store WHERE d_month_seq BETWEEN {dms} AND {dms}+11 AND s_state IN (top_states)
    base = (
        store_sales.join(date_dim, left_on="ss_sold_date_sk", right_on="d_date_sk")
        .join(store, left_on="ss_store_sk", right_on="s_store_sk")
        .join(tmp1, on="s_state", how="semi")
        .filter(pl.col("d_month_seq").is_between(dms, dms + 11))
    )

    all_cols = {"s_state": pl.Utf8, "s_county": pl.Utf8}
    agg_exprs = [pl.col("ss_net_profit").sum().alias("total_sum")]
    output_order = ["total_sum", "s_state", "s_county", "lochierarchy"]

    # SQL: ROLLUP(s_state, s_county) — detail (lochierarchy=0), by_state (=1), total (=2)
    detail = rollup_level(
        base,
        ["s_state", "s_county"],
        all_cols,
        agg_exprs,
        output_order,
        grouping_col="lochierarchy",
        grouping_value=0,
    )
    by_state = rollup_level(
        base,
        ["s_state"],
        all_cols,
        agg_exprs,
        output_order,
        grouping_col="lochierarchy",
        grouping_value=1,
    )
    total = rollup_level(
        base,
        [],
        all_cols,
        agg_exprs,
        output_order,
        grouping_col="lochierarchy",
        grouping_value=2,
    )

    # SQL: UNION ALL of rollup levels
    combined = pl.concat([detail, by_state, total])

    return QueryResult(
        frame=(
            # SQL: CASE WHEN lochierarchy=0 THEN s_state END AS partition_key (for window partition)
            combined.with_columns(
                pl.when(pl.col("lochierarchy") == 0)
                .then(pl.col("s_state"))
                .otherwise(None)
                .alias("partition_key")
            )
            # SQL: Rank() OVER (PARTITION BY lochierarchy+partition_key ORDER BY Sum(ss_net_profit) DESC) AS rank_within_parent
            .with_columns(
                pl.col("total_sum")
                .rank(method="min", descending=True)
                .over(["lochierarchy", "partition_key"])
                .alias("rank_within_parent")
            )
            # SQL: SELECT total_sum, s_state, s_county, lochierarchy, rank_within_parent
            .select(
                [
                    "total_sum",
                    "s_state",
                    "s_county",
                    "lochierarchy",
                    "rank_within_parent",
                ]
            )
            # SQL: ORDER BY lochierarchy DESC, CASE WHEN lochierarchy=0 THEN s_state END, rank_within_parent
            .sort(
                [
                    pl.col("lochierarchy"),
                    pl.when(pl.col("lochierarchy") == 0)
                    .then(pl.col("s_state"))
                    .otherwise(None),
                    pl.col("rank_within_parent"),
                ],
                descending=[True, False, False],
                nulls_last=True,
            )
            # SQL: LIMIT 100
            .limit(100)
        ),
        sort_by=[("lochierarchy", True), ("rank_within_parent", False)],
        limit=100,
        sort_keys=[
            (pl.col("lochierarchy"), True),
            (
                pl.when(pl.col("lochierarchy") == 0)
                .then(pl.col("s_state"))
                .otherwise(None),
                False,
            ),
            (pl.col("rank_within_parent"), False),
        ],
    )
