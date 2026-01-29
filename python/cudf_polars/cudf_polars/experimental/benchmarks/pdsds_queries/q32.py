# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 32."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.utils import get_data

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 32."""
    return """
    SELECT
           Sum(cs_ext_discount_amt) AS 'excess discount amount'
    FROM   catalog_sales ,
           item ,
           date_dim
    WHERE  i_manufact_id = 610
    AND    i_item_sk = cs_item_sk
    AND    d_date BETWEEN '2001-03-04' AND    (
                  Cast('2001-03-04' AS DATE) + INTERVAL '90' day)
    AND    d_date_sk = cs_sold_date_sk
    AND    cs_ext_discount_amt >
           (
                  SELECT 1.3 * avg(cs_ext_discount_amt)
                  FROM   catalog_sales ,
                         date_dim
                  WHERE  cs_item_sk = i_item_sk
                  AND    d_date BETWEEN '2001-03-04' AND    (
                                cast('2001-03-04' AS date) + INTERVAL '90' day)
                  AND    d_date_sk = cs_sold_date_sk )
    LIMIT 100;
    """


def polars_impl(run_config: RunConfig) -> pl.LazyFrame:
    """Query 32."""
    # Load tables
    catalog_sales = get_data(
        run_config.dataset_path, "catalog_sales", run_config.suffix
    )
    item = get_data(run_config.dataset_path, "item", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    # Date range: '2001-03-04' to '2001-03-04' + 90 days = '2001-06-02'
    start_date = pl.date(2001, 3, 4)
    end_date = pl.date(2001, 6, 2)
    # First, calculate the average discount amount for each item in the date range
    item_avg_discounts = (
        catalog_sales.join(date_dim, left_on="cs_sold_date_sk", right_on="d_date_sk")
        .filter(pl.col("d_date").is_between(start_date, end_date))
        .group_by("cs_item_sk")
        .agg([(pl.col("cs_ext_discount_amt").mean() * 1.3).alias("threshold_discount")])
    )
    # Main query: find items with manufacturer 610 and high discount amounts
    return (
        catalog_sales.join(item, left_on="cs_item_sk", right_on="i_item_sk")
        .join(date_dim, left_on="cs_sold_date_sk", right_on="d_date_sk")
        .join(item_avg_discounts, on="cs_item_sk")
        .filter(
            (pl.col("i_manufact_id") == 610)
            & (pl.col("d_date").is_between(start_date, end_date))
            & (pl.col("cs_ext_discount_amt") > pl.col("threshold_discount"))
        )
        .select([pl.col("cs_ext_discount_amt").sum().alias("excess discount amount")])
        .limit(100)
    )
