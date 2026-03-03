# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 20."""

from __future__ import annotations

from datetime import date, timedelta
from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.pdsds_parameters import load_parameters
from cudf_polars.experimental.benchmarks.utils import get_data

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 20."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=20,
        qualification=run_config.qualification,
    )

    sdate = params["sdate"]
    categories = params["category"]

    return f"""
    SELECT
             i_item_id ,
             i_item_desc ,
             i_category ,
             i_class ,
             i_current_price ,
             Sum(cs_ext_sales_price)                                                              AS itemrevenue ,
             Sum(cs_ext_sales_price)*100/Sum(Sum(cs_ext_sales_price)) OVER (partition BY i_class) AS revenueratio
    FROM     catalog_sales ,
             item ,
             date_dim
    WHERE    cs_item_sk = i_item_sk
    AND      i_category IN ({", ".join(f"'{cat}'" for cat in categories)})
    AND      cs_sold_date_sk = d_date_sk
    AND      d_date BETWEEN Cast('{sdate}' AS DATE) AND      (
                      Cast('{sdate}' AS DATE) + INTERVAL '30' day)
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
    """Query 20."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=20,
        qualification=run_config.qualification,
    )

    sdate = params["sdate"]
    categories = params["category"]

    # Load tables
    catalog_sales = get_data(
        run_config.dataset_path, "catalog_sales", run_config.suffix
    )
    item = get_data(run_config.dataset_path, "item", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)

    # Parse sdate and compute end date
    start_date = date.fromisoformat(sdate)
    end_date = start_date + timedelta(days=30)

    # Convert to string literals for comparison (d_date is String in parquet)
    start_date_str = pl.lit(start_date.isoformat())
    end_date_str = pl.lit(end_date.isoformat())
    return (
        catalog_sales.join(item, left_on="cs_item_sk", right_on="i_item_sk")
        .join(date_dim, left_on="cs_sold_date_sk", right_on="d_date_sk")
        .filter(
            pl.col("i_category").is_in(categories)
            & pl.col("d_date").is_between(start_date_str, end_date_str, closed="both")
        )
        .group_by(
            ["i_item_id", "i_item_desc", "i_category", "i_class", "i_current_price"]
        )
        .agg([pl.col("cs_ext_sales_price").sum().alias("itemrevenue")])
        .with_columns(
            [
                # Handle case where itemrevenue is 0 - should result in NULL like SQL
                pl.when(pl.col("itemrevenue") == 0.0)
                .then(None)
                .otherwise(
                    pl.col("itemrevenue")
                    * 100
                    / pl.col("itemrevenue").sum().over("i_class")
                )
                .alias("revenueratio")
            ]
        )
        .sort(
            ["i_category", "i_class", "i_item_id", "i_item_desc", "revenueratio"],
            nulls_last=True,
        )
        .limit(100)
    )
