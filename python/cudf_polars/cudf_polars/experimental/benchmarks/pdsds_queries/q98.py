# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 98."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.utils import get_data

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 98."""
    return """
    -- start query 98 in stream 0 using template query98.tpl
    SELECT i_item_id,
           i_item_desc,
           i_category,
           i_class,
           i_current_price,
           Sum(ss_ext_sales_price)                                   AS itemrevenue,
           Sum(ss_ext_sales_price) * 100 / Sum(Sum(ss_ext_sales_price))
                                             OVER (
                                               PARTITION BY i_class) AS revenueratio
    FROM   store_sales,
           item,
           date_dim
    WHERE  ss_item_sk = i_item_sk
           AND i_category IN ( 'Men', 'Home', 'Electronics' )
           AND ss_sold_date_sk = d_date_sk
           AND d_date BETWEEN CAST('2000-05-18' AS DATE) AND (
                              CAST('2000-05-18' AS DATE) + INTERVAL '30' DAY )
    GROUP  BY i_item_id,
              i_item_desc,
              i_category,
              i_class,
              i_current_price
    ORDER  BY i_category,
              i_class,
              i_item_id,
              i_item_desc,
              revenueratio;
    """


def polars_impl(run_config: RunConfig) -> pl.LazyFrame:
    """Query 98."""
    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    item = get_data(run_config.dataset_path, "item", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    start_date = pl.date(2000, 5, 18)
    end_date = pl.date(2000, 6, 17)
    return (
        store_sales.join(item, left_on="ss_item_sk", right_on="i_item_sk", how="inner")
        .join(date_dim, left_on="ss_sold_date_sk", right_on="d_date_sk", how="inner")
        .filter(
            pl.col("i_category").is_in(["Men", "Home", "Electronics"])
            & pl.col("d_date").is_between(start_date, end_date, closed="both")
        )
        .group_by(
            ["i_item_id", "i_item_desc", "i_category", "i_class", "i_current_price"]
        )
        .agg(
            [
                pl.col("ss_ext_sales_price").count().alias("itemrevenue_count"),
                pl.col("ss_ext_sales_price").sum().alias("itemrevenue_sum"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("itemrevenue_count") == 0)
                .then(None)
                .otherwise(pl.col("itemrevenue_sum"))
                .alias("itemrevenue")
            ]
        )
        .with_columns(
            [
                (
                    pl.col("itemrevenue")
                    * 100.0
                    / pl.col("itemrevenue").sum().over("i_class")
                ).alias("revenueratio")
            ]
        )
        .select(
            [
                "i_item_id",
                "i_item_desc",
                "i_category",
                "i_class",
                "i_current_price",
                "itemrevenue",
                "revenueratio",
            ]
        )
        .sort(
            ["i_category", "i_class", "i_item_id", "i_item_desc", "revenueratio"],
            nulls_last=True,
        )
    )
