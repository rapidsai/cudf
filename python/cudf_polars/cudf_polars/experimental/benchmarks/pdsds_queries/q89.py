# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 89."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.utils import get_data

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 89."""
    return """
    SELECT  *
    FROM  (SELECT i_category,
                  i_class,
                  i_brand,
                  s_store_name,
                  s_company_name,
                  d_moy,
                  Sum(ss_sales_price) sum_sales,
                  Avg(Sum(ss_sales_price))
                    OVER (
                      partition BY i_category, i_brand, s_store_name, s_company_name
                    )
                                      avg_monthly_sales
           FROM   item,
                  store_sales,
                  date_dim,
                  store
           WHERE  ss_item_sk = i_item_sk
                  AND ss_sold_date_sk = d_date_sk
                  AND ss_store_sk = s_store_sk
                  AND d_year IN ( 2002 )
                  AND ( ( i_category IN ( 'Home', 'Men', 'Sports' )
                          AND i_class IN ( 'paint', 'accessories', 'fitness' ) )
                         OR ( i_category IN ( 'Shoes', 'Jewelry', 'Women' )
                              AND i_class IN ( 'mens', 'pendants', 'swimwear' ) ) )
           GROUP  BY i_category,
                     i_class,
                     i_brand,
                     s_store_name,
                     s_company_name,
                     d_moy) tmp1
    WHERE  CASE
             WHEN ( avg_monthly_sales <> 0 ) THEN (
             Abs(sum_sales - avg_monthly_sales) / avg_monthly_sales )
             ELSE NULL
           END > 0.1
    ORDER  BY sum_sales - avg_monthly_sales,
              s_store_name
    LIMIT 100;
    """


def polars_impl(run_config: RunConfig) -> pl.LazyFrame:
    """Query 89."""
    item = get_data(run_config.dataset_path, "item", run_config.suffix)
    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    store = get_data(run_config.dataset_path, "store", run_config.suffix)

    filter1 = (pl.col("i_category").is_in(["Home", "Men", "Sports"])) & (
        pl.col("i_class").is_in(["paint", "accessories", "fitness"])
    )
    filter2 = (pl.col("i_category").is_in(["Shoes", "Jewelry", "Women"])) & (
        pl.col("i_class").is_in(["mens", "pendants", "swimwear"])
    )

    tmp1 = (
        store_sales.join(item, left_on="ss_item_sk", right_on="i_item_sk", how="inner")
        .join(date_dim, left_on="ss_sold_date_sk", right_on="d_date_sk", how="inner")
        .join(store, left_on="ss_store_sk", right_on="s_store_sk", how="inner")
        .filter((pl.col("d_year") == 2002) & (filter1 | filter2))
        .group_by(
            [
                "i_category",
                "i_class",
                "i_brand",
                "s_store_name",
                "s_company_name",
                "d_moy",
            ]
        )
        .agg([pl.col("ss_sales_price").sum().alias("sum_sales")])
        .with_columns(
            pl.col("sum_sales")
            .mean()
            .over(["i_category", "i_brand", "s_store_name", "s_company_name"])
            .alias("avg_monthly_sales")
        )
    )

    return (
        tmp1.with_columns(
            pl.when(pl.col("avg_monthly_sales") != 0)
            .then(
                (pl.col("sum_sales") - pl.col("avg_monthly_sales")).abs()
                / pl.col("avg_monthly_sales")
            )
            .otherwise(None)
            .alias("deviation_ratio")
        )
        .filter(pl.col("deviation_ratio") > 0.1)
        .select(
            [
                "i_category",
                "i_class",
                "i_brand",
                "s_store_name",
                "s_company_name",
                "d_moy",
                "sum_sales",
                "avg_monthly_sales",
            ]
        )
        .sort(
            [(pl.col("sum_sales") - pl.col("avg_monthly_sales")), "s_store_name"],
            nulls_last=True,
        )
        .limit(100)
    )
