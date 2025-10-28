# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 55."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.utils import get_data

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 55."""
    return """
    SELECT i_brand_id              brand_id, 
                   i_brand                 brand, 
                   Sum(ss_ext_sales_price) ext_price 
    FROM   date_dim, 
           store_sales, 
           item 
    WHERE  d_date_sk = ss_sold_date_sk 
           AND ss_item_sk = i_item_sk 
           AND i_manager_id = 33 
           AND d_moy = 12 
           AND d_year = 1998 
    GROUP  BY i_brand, 
              i_brand_id 
    ORDER  BY ext_price DESC, 
              i_brand_id
    LIMIT 100;
    """


def polars_impl(run_config: RunConfig) -> pl.LazyFrame:
    """Query 55."""
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    item = get_data(run_config.dataset_path, "item", run_config.suffix)
    return (
        store_sales
        .join(date_dim, left_on="ss_sold_date_sk", right_on="d_date_sk")
        .join(item, left_on="ss_item_sk", right_on="i_item_sk")
        .filter(
            (pl.col("i_manager_id") == 33)
            & (pl.col("d_moy") == 12)
            & (pl.col("d_year") == 1998)
        )
        .group_by(["i_brand", "i_brand_id"])
        .agg(pl.col("ss_ext_sales_price").sum().alias("ext_price"))
        .select([
            pl.col("i_brand_id").alias("brand_id"),
            pl.col("i_brand").alias("brand"),
            pl.col("ext_price"),
        ])
        .sort(["ext_price", "brand_id"], descending=[True, False], nulls_last=True)
        .limit(100)
    )
