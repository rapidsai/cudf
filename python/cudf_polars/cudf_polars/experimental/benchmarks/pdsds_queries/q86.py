# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 86."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.utils import get_data

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 86."""
    return """
    SELECT Sum(ws_net_paid)                         AS total_sum,
                   i_category,
                   i_class,
                   Grouping(i_category) + Grouping(i_class) AS lochierarchy,
                   Rank()
                     OVER (
                       partition BY Grouping(i_category)+Grouping(i_class), CASE
                     WHEN Grouping(
                     i_class) = 0 THEN i_category END
                       ORDER BY Sum(ws_net_paid) DESC)      AS rank_within_parent
    FROM   web_sales,
           date_dim d1,
           item
    WHERE  d1.d_month_seq BETWEEN 1183 AND 1183 + 11
           AND d1.d_date_sk = ws_sold_date_sk
           AND i_item_sk = ws_item_sk
    GROUP  BY rollup( i_category, i_class )
    ORDER  BY lochierarchy DESC,
              CASE
                WHEN lochierarchy = 0 THEN i_category
              END,
              rank_within_parent
    LIMIT 100;
    """


def _rollup_level(
    base: pl.LazyFrame, group_cols: list[str] | None, lochierarchy: int
) -> pl.LazyFrame:
    if group_cols:
        out = (
            base.group_by(group_cols)
            .agg(pl.col("ws_net_paid").sum().alias("total_sum"))
            .with_columns(pl.lit(lochierarchy, dtype=pl.Int64).alias("lochierarchy"))
        )
        if group_cols == ["i_category", "i_class"]:
            out = out.select(["total_sum", "i_category", "i_class", "lochierarchy"])
        elif group_cols == ["i_category"]:
            out = out.select(
                [
                    "total_sum",
                    "i_category",
                    pl.lit(None).cast(pl.Utf8).alias("i_class"),
                    "lochierarchy",
                ]
            )
        else:
            out = out.select(
                [
                    "total_sum",
                    pl.lit(None).cast(pl.Utf8).alias("i_category"),
                    pl.lit(None).cast(pl.Utf8).alias("i_class"),
                    "lochierarchy",
                ]
            )
    else:
        out = (
            base.select(pl.col("ws_net_paid").sum().alias("total_sum"))
            .with_columns(pl.lit(lochierarchy, dtype=pl.Int64).alias("lochierarchy"))
            .select(
                [
                    "total_sum",
                    pl.lit(None).cast(pl.Utf8).alias("i_category"),
                    pl.lit(None).cast(pl.Utf8).alias("i_class"),
                    "lochierarchy",
                ]
            )
        )
    return out


def polars_impl(run_config: RunConfig) -> pl.LazyFrame:
    """Query 86."""
    web_sales = get_data(run_config.dataset_path, "web_sales", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    item = get_data(run_config.dataset_path, "item", run_config.suffix)

    base = (
        web_sales.join(
            date_dim.filter(pl.col("d_month_seq").is_between(1183, 1183 + 11)).select(
                ["d_date_sk"]
            ),
            left_on="ws_sold_date_sk",
            right_on="d_date_sk",
        )
        .join(item, left_on="ws_item_sk", right_on="i_item_sk")
        .select(["ws_net_paid", "i_category", "i_class"])
    )

    lvl0 = _rollup_level(base, ["i_category", "i_class"], 0)
    lvl1 = _rollup_level(base, ["i_category"], 1)
    lvl2 = _rollup_level(base, None, 2)

    combined = pl.concat([lvl0, lvl1, lvl2])

    return (
        combined.with_columns(
            pl.when(pl.col("lochierarchy") == 0)
            .then(pl.col("i_category"))
            .otherwise(None)
            .alias("partition_category")
        )
        .with_columns(
            pl.col("total_sum")
            .rank(method="min", descending=True)
            .over([pl.col("lochierarchy"), pl.col("partition_category")])
            .cast(pl.Int64)
            .alias("rank_within_parent")
        )
        .select(
            ["total_sum", "i_category", "i_class", "lochierarchy", "rank_within_parent"]
        )
        .sort(
            [
                "lochierarchy",
                pl.when(pl.col("lochierarchy") == 0)
                .then(pl.col("i_category"))
                .otherwise(None),
                "rank_within_parent",
            ],
            descending=[True, False, False],
            nulls_last=True,
        )
        .limit(100)
    )
