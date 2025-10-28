# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 63."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.utils import get_data

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 63."""
    return """
    SELECT * 
    FROM   (SELECT i_manager_id, 
                   Sum(ss_sales_price)            sum_sales, 
                   Avg(Sum(ss_sales_price)) 
                     OVER ( 
                       partition BY i_manager_id) avg_monthly_sales 
            FROM   item, 
                   store_sales, 
                   date_dim, 
                   store 
            WHERE  ss_item_sk = i_item_sk 
                   AND ss_sold_date_sk = d_date_sk 
                   AND ss_store_sk = s_store_sk 
                   AND d_month_seq IN ( 1200, 1200 + 1, 1200 + 2, 1200 + 3, 
                                        1200 + 4, 1200 + 5, 1200 + 6, 1200 + 7, 
                                        1200 + 8, 1200 + 9, 1200 + 10, 1200 + 11 ) 
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
            GROUP  BY i_manager_id, 
                      d_moy) tmp1 
    WHERE  CASE 
             WHEN avg_monthly_sales > 0 THEN Abs (sum_sales - avg_monthly_sales) / 
                                             avg_monthly_sales 
             ELSE NULL 
           END > 0.1 
    ORDER  BY i_manager_id, 
              avg_monthly_sales, 
              sum_sales
    LIMIT 100;
    """


def polars_impl(run_config: RunConfig) -> pl.LazyFrame:
    """Query 63."""
    item = get_data(run_config.dataset_path, "item", run_config.suffix)
    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    store = get_data(run_config.dataset_path, "store", run_config.suffix)

    inner_query = (
        store_sales
        .join(item, left_on="ss_item_sk", right_on="i_item_sk")
        .join(date_dim, left_on="ss_sold_date_sk", right_on="d_date_sk")
        .join(store, left_on="ss_store_sk", right_on="s_store_sk")
        .filter(
            pl.col("d_month_seq").is_in([
                1200, 1201, 1202, 1203, 1204, 1205,
                1206, 1207, 1208, 1209, 1210, 1211
            ]) &
            (
                (
                    pl.col("i_category").is_in(["Books", "Children", "Electronics"]) &
                    pl.col("i_class").is_in(["personal", "portable", "reference", "self-help"]) &
                    pl.col("i_brand").is_in([
                        "scholaramalgamalg #14", "scholaramalgamalg #7",
                        "exportiunivamalg #9", "scholaramalgamalg #9"
                    ])
                ) |
                (
                    pl.col("i_category").is_in(["Women", "Music", "Men"]) &
                    pl.col("i_class").is_in(["accessories", "classical", "fragrances", "pants"]) &
                    pl.col("i_brand").is_in([
                        "amalgimporto #1", "edu packscholar #1",
                        "exportiimporto #1", "importoamalg #1"
                    ])
                )
            )
        )
        .group_by(["i_manager_id", "d_moy"])
        .agg([pl.col("ss_sales_price").sum().alias("sum_sales")])
        .with_columns(
            pl.col("sum_sales").mean().over("i_manager_id").alias("avg_monthly_sales")
        )
    )

    return (
        inner_query
        .with_columns(
            pl.when(pl.col("avg_monthly_sales") > 0)
              .then((pl.col("sum_sales") - pl.col("avg_monthly_sales")).abs() / pl.col("avg_monthly_sales"))
              .otherwise(None)
              .alias("deviation")
        )
        .filter(pl.col("deviation") > 0.1)
        .select(["i_manager_id", "sum_sales", "avg_monthly_sales"])
        .sort(["i_manager_id", "avg_monthly_sales", "sum_sales"], nulls_last=True, descending=[False, False, False])
        .limit(100)
    )
