# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 53."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.utils import get_data

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 53."""
    return """
    SELECT *
    FROM   (SELECT i_manufact_id,
                   Sum(ss_sales_price)             sum_sales,
                   Avg(Sum(ss_sales_price))
                     OVER (
                       partition BY i_manufact_id) avg_quarterly_sales
            FROM   item,
                   store_sales,
                   date_dim,
                   store
            WHERE  ss_item_sk = i_item_sk
                   AND ss_sold_date_sk = d_date_sk
                   AND ss_store_sk = s_store_sk
                   AND d_month_seq IN ( 1199, 1199 + 1, 1199 + 2, 1199 + 3,
                                        1199 + 4, 1199 + 5, 1199 + 6, 1199 + 7,
                                        1199 + 8, 1199 + 9, 1199 + 10, 1199 + 11 )
                   AND ( ( i_category IN ( 'Books', 'Children', 'Electronics' )
                           AND i_class IN ( 'personal', 'portable', 'reference',
                                            'self-help' )
                           AND i_brand IN ( 'scholaramalgamalg #14',
                                            'scholaramalgamalg #7'
                                            ,
                                            'exportiunivamalg #9',
                                                           'scholaramalgamalg #9' )
                         )
                          OR ( i_category IN ( 'Women', 'Music', 'Men' )
                               AND i_class IN ( 'accessories', 'classical',
                                                'fragrances',
                                                'pants' )
                               AND i_brand IN ( 'amalgimporto #1',
                                                'edu packscholar #1',
                                                'exportiimporto #1',
                                                    'importoamalg #1' ) ) )
            GROUP  BY i_manufact_id,
                      d_qoy) tmp1
    WHERE  CASE
             WHEN avg_quarterly_sales > 0 THEN Abs (sum_sales - avg_quarterly_sales)
                                               /
                                               avg_quarterly_sales
             ELSE NULL
           END > 0.1
    ORDER  BY avg_quarterly_sales,
              sum_sales,
              i_manufact_id
    LIMIT 100;
    """


def polars_impl(run_config: RunConfig) -> pl.LazyFrame:
    """Query 53."""
    # Load tables
    item = get_data(run_config.dataset_path, "item", run_config.suffix)
    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    store = get_data(run_config.dataset_path, "store", run_config.suffix)
    month_seq_list = list(range(1199, 1199 + 12))
    grouped_data = (
        store_sales.join(item, left_on="ss_item_sk", right_on="i_item_sk")
        .join(date_dim, left_on="ss_sold_date_sk", right_on="d_date_sk")
        .join(store, left_on="ss_store_sk", right_on="s_store_sk")
        .filter(pl.col("d_month_seq").is_in(month_seq_list))
        .filter(
            # Books/Children/Electronics categories
            (
                (pl.col("i_category").is_in(["Books", "Children", "Electronics"]))
                & (
                    pl.col("i_class").is_in(
                        ["personal", "portable", "reference", "self-help"]
                    )
                )
                & (
                    pl.col("i_brand").is_in(
                        [
                            "scholaramalgamalg #14",
                            "scholaramalgamalg #7",
                            "exportiunivamalg #9",
                            "scholaramalgamalg #9",
                        ]
                    )
                )
            )
            |
            # Women/Music/Men categories
            (
                (pl.col("i_category").is_in(["Women", "Music", "Men"]))
                & (
                    pl.col("i_class").is_in(
                        ["accessories", "classical", "fragrances", "pants"]
                    )
                )
                & (
                    pl.col("i_brand").is_in(
                        [
                            "amalgimporto #1",
                            "edu packscholar #1",
                            "exportiimporto #1",
                            "importoamalg #1",
                        ]
                    )
                )
            )
        )
        .group_by(["i_manufact_id", "d_qoy"])
        .agg([pl.col("ss_sales_price").sum().alias("sum_sales_raw")])
        .with_columns(
            [
                pl.when(pl.col("sum_sales_raw").is_not_null())
                .then(pl.col("sum_sales_raw"))
                .otherwise(None)
                .alias("sum(ss_sales_price)")
            ]
        )
    )
    non_null_data = grouped_data.filter(pl.col("i_manufact_id").is_not_null())
    null_data = grouped_data.filter(pl.col("i_manufact_id").is_null())
    manufacturer_averages = non_null_data.group_by("i_manufact_id").agg(
        [pl.col("sum(ss_sales_price)").mean().alias("avg_quarterly_sales")]
    )
    non_null_result = non_null_data.join(
        manufacturer_averages, on="i_manufact_id", how="left"
    )
    null_result = null_data.with_columns(
        [pl.col("sum(ss_sales_price)").mean().alias("avg_quarterly_sales")]
    )
    inner_query = pl.concat([non_null_result, null_result]).select(
        [
            "i_manufact_id",
            pl.col("sum(ss_sales_price)").alias("sum_sales"),
            "avg_quarterly_sales",
        ]
    )
    return (
        inner_query.filter(
            # Percentage deviation > 10%
            pl.when(pl.col("avg_quarterly_sales") > 0)
            .then(
                (pl.col("sum_sales") - pl.col("avg_quarterly_sales")).abs()
                / pl.col("avg_quarterly_sales")
            )
            .otherwise(None)
            > 0.1
        )
        .select(["i_manufact_id", "sum_sales", "avg_quarterly_sales"])
        .sort(
            ["avg_quarterly_sales", "sum_sales", "i_manufact_id"],
            nulls_last=True,
            descending=[False, False, False],
        )
        .limit(100)
    )
