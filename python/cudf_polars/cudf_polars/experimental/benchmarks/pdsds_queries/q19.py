# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 19."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.utils import get_data

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 19."""
    return """
    SELECT i_brand_id              brand_id,
                   i_brand                 brand,
                   i_manufact_id,
                   i_manufact,
                   Sum(ss_ext_sales_price) ext_price
    FROM   date_dim,
           store_sales,
           item,
           customer,
           customer_address,
           store
    WHERE  d_date_sk = ss_sold_date_sk
           AND ss_item_sk = i_item_sk
           AND i_manager_id = 38
           AND d_moy = 12
           AND d_year = 1998
           AND ss_customer_sk = c_customer_sk
           AND c_current_addr_sk = ca_address_sk
           AND Substr(ca_zip, 1, 5) <> Substr(s_zip, 1, 5)
           AND ss_store_sk = s_store_sk
    GROUP  BY i_brand,
              i_brand_id,
              i_manufact_id,
              i_manufact
    ORDER  BY ext_price DESC,
              i_brand,
              i_brand_id,
              i_manufact_id,
              i_manufact
    LIMIT 100;
    """


def polars_impl(run_config: RunConfig) -> pl.LazyFrame:
    """Query 19."""
    # Load tables
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    item = get_data(run_config.dataset_path, "item", run_config.suffix)
    customer = get_data(run_config.dataset_path, "customer", run_config.suffix)
    customer_address = get_data(
        run_config.dataset_path, "customer_address", run_config.suffix
    )
    store = get_data(run_config.dataset_path, "store", run_config.suffix)
    return (
        date_dim.join(store_sales, left_on="d_date_sk", right_on="ss_sold_date_sk")
        .join(item, left_on="ss_item_sk", right_on="i_item_sk")
        .join(customer, left_on="ss_customer_sk", right_on="c_customer_sk")
        .join(customer_address, left_on="c_current_addr_sk", right_on="ca_address_sk")
        .join(store, left_on="ss_store_sk", right_on="s_store_sk")
        .filter(
            (pl.col("i_manager_id") == 38)
            & (pl.col("d_moy") == 12)
            & (pl.col("d_year") == 1998)
            & (pl.col("ca_zip").str.slice(0, 5) != pl.col("s_zip").str.slice(0, 5))
        )
        .group_by(["i_brand", "i_brand_id", "i_manufact_id", "i_manufact"])
        .agg([pl.col("ss_ext_sales_price").sum().alias("ext_price")])
        .select(
            [
                pl.col("i_brand_id").alias("brand_id"),
                pl.col("i_brand").alias("brand"),
                "i_manufact_id",
                "i_manufact",
                "ext_price",
            ]
        )
        .sort(
            ["ext_price", "brand", "brand_id", "i_manufact_id", "i_manufact"],
            descending=[True, False, False, False, False],
            nulls_last=True,
        )
        .limit(100)
    )
