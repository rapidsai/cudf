# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 45."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.utils import get_data

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 45."""
    return """
    SELECT ca_zip,
                   ca_state,
                   Sum(ws_sales_price)
    FROM   web_sales,
           customer,
           customer_address,
           date_dim,
           item
    WHERE  ws_bill_customer_sk = c_customer_sk
           AND c_current_addr_sk = ca_address_sk
           AND ws_item_sk = i_item_sk
           AND ( Substr(ca_zip, 1, 5) IN ( '85669', '86197', '88274', '83405',
                                           '86475', '85392', '85460', '80348',
                                           '81792' )
                  OR i_item_id IN (SELECT i_item_id
                                   FROM   item
                                   WHERE  i_item_sk IN ( 2, 3, 5, 7,
                                                         11, 13, 17, 19,
                                                         23, 29 )) )
           AND ws_sold_date_sk = d_date_sk
           AND d_qoy = 1
           AND d_year = 2000
    GROUP  BY ca_zip,
              ca_state
    ORDER  BY ca_zip,
              ca_state
    LIMIT 100;
    """


def polars_impl(run_config: RunConfig) -> pl.LazyFrame:
    """Query 45."""
    # Load tables
    web_sales = get_data(run_config.dataset_path, "web_sales", run_config.suffix)
    customer = get_data(run_config.dataset_path, "customer", run_config.suffix)
    customer_address = get_data(
        run_config.dataset_path, "customer_address", run_config.suffix
    )
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    item = get_data(run_config.dataset_path, "item", run_config.suffix)

    # Subquery: filter item table to just those i_item_id matching i_item_sk
    filtered_items = (
        item.filter(pl.col("i_item_sk").is_in([2, 3, 5, 7, 11, 13, 17, 19, 23, 29]))
        .select("i_item_id")
        .unique()
    )

    # Perform all joins first and filter for date
    joined = (
        web_sales.join(
            customer, left_on="ws_bill_customer_sk", right_on="c_customer_sk"
        )
        .join(customer_address, left_on="c_current_addr_sk", right_on="ca_address_sk")
        .join(item, left_on="ws_item_sk", right_on="i_item_sk")
        .join(date_dim, left_on="ws_sold_date_sk", right_on="d_date_sk")
        .filter((pl.col("d_qoy") == 1) & (pl.col("d_year") == 2000))
        # Extract first 5 characters of ZIP code
        .with_columns([pl.col("ca_zip").str.slice(0, 5).alias("zip_prefix")])
    )

    # First condition: zip code prefix in target list
    zip_match = joined.filter(
        pl.col("zip_prefix").is_in(
            [
                "85669",
                "86197",
                "88274",
                "83405",
                "86475",
                "85392",
                "85460",
                "80348",
                "81792",
            ]
        )
    )

    # Second condition: item ID in filtered subquery result
    item_match = joined.join(
        filtered_items, left_on="i_item_id", right_on="i_item_id", how="semi"
    )

    return (
        pl.concat([zip_match, item_match])
        .group_by(["ca_zip", "ca_state"])
        .agg(
            [
                pl.col("ws_sales_price").sum().alias("sales_sum"),
                pl.col("ws_sales_price").count().alias("sales_count"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("sales_count") > 0)
                .then(pl.col("sales_sum"))
                .otherwise(None)
                .alias("sum(ws_sales_price)")
            ]
        )
        .drop(["sales_sum", "sales_count"])
        .sort(["ca_zip", "ca_state"], nulls_last=True)
        .select(["ca_zip", "ca_state", "sum(ws_sales_price)"])
        .limit(100)
    )
