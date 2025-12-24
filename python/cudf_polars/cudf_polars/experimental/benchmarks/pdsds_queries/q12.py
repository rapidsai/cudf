# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 12."""

from __future__ import annotations

from datetime import date, timedelta
from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.utils import get_data

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 12."""
    return """
    SELECT
             i_item_id ,
             i_item_desc ,
             i_category ,
             i_class ,
             i_current_price ,
             Sum(ws_ext_sales_price)                                                              AS itemrevenue ,
             Sum(ws_ext_sales_price)*100/Sum(Sum(ws_ext_sales_price)) OVER (partition BY i_class) AS revenueratio
    FROM     web_sales ,
             item ,
             date_dim
    WHERE    ws_item_sk = i_item_sk
    AND      i_category IN ('Home',
                            'Men',
                            'Women')
    AND      ws_sold_date_sk = d_date_sk
    AND      d_date BETWEEN Cast('2000-05-11' AS DATE) AND      (
                      Cast('2000-05-11' AS DATE) + INTERVAL '30' day)
    GROUP BY i_item_id ,
             i_item_desc ,
             i_category ,
             i_class ,
             i_current_price
    ORDER BY i_category ,
             i_class ,
             i_item_id ,
             i_item_desc ,
             revenueratio
    LIMIT 100;
    """


def polars_impl(run_config: RunConfig) -> pl.LazyFrame:
    """Query 12."""
    # Load tables
    web_sales = get_data(run_config.dataset_path, "web_sales", run_config.suffix)
    item = get_data(run_config.dataset_path, "item", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    start_date = date(2000, 5, 11)
    end_date = start_date + timedelta(days=30)
    return (
        web_sales.join(item, left_on="ws_item_sk", right_on="i_item_sk")
        .join(date_dim, left_on="ws_sold_date_sk", right_on="d_date_sk")
        .filter(
            pl.col("i_category").is_in(["Home", "Men", "Women"])
            & pl.col("d_date").is_between(start_date, end_date, closed="both")
        )
        .group_by(
            ["i_item_id", "i_item_desc", "i_category", "i_class", "i_current_price"]
        )
        .agg(
            [
                pl.when(pl.col("ws_ext_sales_price").count() > 0)
                .then(pl.col("ws_ext_sales_price").sum())
                .otherwise(None)
                .alias("itemrevenue")
            ]
        )
        .with_columns(
            [
                (
                    pl.col("itemrevenue")
                    * 100
                    / pl.when(pl.col("itemrevenue").count() > 0)
                    .then(pl.col("itemrevenue").sum())
                    .otherwise(None)
                    .over("i_class")
                ).alias("revenueratio")
            ]
        )
        .sort(
            ["i_category", "i_class", "i_item_id", "i_item_desc", "revenueratio"],
            nulls_last=True,
        )
        .limit(100)
    )
