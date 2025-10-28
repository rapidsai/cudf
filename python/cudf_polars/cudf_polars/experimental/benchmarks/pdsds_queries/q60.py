# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 60."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.utils import get_data

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 60."""
    return """
    WITH ss 
         AS (SELECT i_item_id, 
                    Sum(ss_ext_sales_price) total_sales 
             FROM   store_sales, 
                    date_dim, 
                    customer_address, 
                    item 
             WHERE  i_item_id IN (SELECT i_item_id 
                                  FROM   item 
                                  WHERE  i_category IN ( 'Jewelry' )) 
                    AND ss_item_sk = i_item_sk 
                    AND ss_sold_date_sk = d_date_sk 
                    AND d_year = 1999 
                    AND d_moy = 8 
                    AND ss_addr_sk = ca_address_sk 
                    AND ca_gmt_offset = -6 
             GROUP  BY i_item_id), 
         cs 
         AS (SELECT i_item_id, 
                    Sum(cs_ext_sales_price) total_sales 
             FROM   catalog_sales, 
                    date_dim, 
                    customer_address, 
                    item 
             WHERE  i_item_id IN (SELECT i_item_id 
                                  FROM   item 
                                  WHERE  i_category IN ( 'Jewelry' )) 
                    AND cs_item_sk = i_item_sk 
                    AND cs_sold_date_sk = d_date_sk 
                    AND d_year = 1999 
                    AND d_moy = 8 
                    AND cs_bill_addr_sk = ca_address_sk 
                    AND ca_gmt_offset = -6 
             GROUP  BY i_item_id), 
         ws 
         AS (SELECT i_item_id, 
                    Sum(ws_ext_sales_price) total_sales 
             FROM   web_sales, 
                    date_dim, 
                    customer_address, 
                    item 
             WHERE  i_item_id IN (SELECT i_item_id 
                                  FROM   item 
                                  WHERE  i_category IN ( 'Jewelry' )) 
                    AND ws_item_sk = i_item_sk 
                    AND ws_sold_date_sk = d_date_sk 
                    AND d_year = 1999 
                    AND d_moy = 8 
                    AND ws_bill_addr_sk = ca_address_sk 
                    AND ca_gmt_offset = -6 
             GROUP  BY i_item_id) 
    SELECT i_item_id, 
                   Sum(total_sales) total_sales 
    FROM   (SELECT * 
            FROM   ss 
            UNION ALL 
            SELECT * 
            FROM   cs 
            UNION ALL 
            SELECT * 
            FROM   ws) tmp1 
    GROUP  BY i_item_id 
    ORDER  BY i_item_id, 
              total_sales
    LIMIT 100;
    """


def polars_impl(run_config: RunConfig) -> pl.LazyFrame:
    """Query 60."""
    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    catalog_sales = get_data(run_config.dataset_path, "catalog_sales", run_config.suffix)
    web_sales = get_data(run_config.dataset_path, "web_sales", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    customer_address = get_data(run_config.dataset_path, "customer_address", run_config.suffix)
    item = get_data(run_config.dataset_path, "item", run_config.suffix)

    jewelry_item_ids_lf = (
        item
        .filter(pl.col("i_category") == "Jewelry")
        .select(["i_item_id"])
        .unique()
    )

    channels = [
        ("ss", store_sales,  "ss_item_sk", "ss_sold_date_sk", "ss_addr_sk",   "ss_ext_sales_price"),
        ("cs", catalog_sales,"cs_item_sk", "cs_sold_date_sk", "cs_bill_addr_sk","cs_ext_sales_price"),
        ("ws", web_sales,    "ws_item_sk", "ws_sold_date_sk", "ws_bill_addr_sk","ws_ext_sales_price"),
    ]

    parts: list[pl.LazyFrame] = []
    for _, sales_lf, item_fk, date_fk, addr_fk, price_col in channels:
        parts.append(
            sales_lf
            .join(item, left_on=item_fk, right_on="i_item_sk")
            .join(jewelry_item_ids_lf, on="i_item_id")
            .join(date_dim, left_on=date_fk, right_on="d_date_sk")
            .join(customer_address, left_on=addr_fk, right_on="ca_address_sk")
            .filter(
                (pl.col("d_year") == 1999)
                & (pl.col("d_moy") == 8)
                & (pl.col("ca_gmt_offset") == -6)
            )
            .group_by("i_item_id")
            .agg([
                pl.col(price_col).sum().alias("total_sales")
            ])
            .select(["i_item_id", "total_sales"])
        )

    return (
        pl.concat(parts)
        .group_by("i_item_id")
        .agg([
            pl.col("total_sales").sum().alias("total_sales")
        ])
        .sort(["i_item_id", "total_sales"], descending=[False, False], nulls_last=True)
        .limit(100)
    )
