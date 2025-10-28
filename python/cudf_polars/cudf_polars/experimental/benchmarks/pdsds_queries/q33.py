# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 33."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.utils import get_data

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 33."""
    return """
    WITH ss
         AS (SELECT i_manufact_id,
                    Sum(ss_ext_sales_price) total_sales
             FROM   store_sales,
                    date_dim,
                    customer_address,
                    item
             WHERE  i_manufact_id IN (SELECT i_manufact_id
                                      FROM   item
                                      WHERE  i_category IN ( 'Books' ))
                    AND ss_item_sk = i_item_sk
                    AND ss_sold_date_sk = d_date_sk
                    AND d_year = 1999
                    AND d_moy = 3
                    AND ss_addr_sk = ca_address_sk
                    AND ca_gmt_offset = -5
             GROUP  BY i_manufact_id),
         cs
         AS (SELECT i_manufact_id,
                    Sum(cs_ext_sales_price) total_sales
             FROM   catalog_sales,
                    date_dim,
                    customer_address,
                    item
             WHERE  i_manufact_id IN (SELECT i_manufact_id
                                      FROM   item
                                      WHERE  i_category IN ( 'Books' ))
                    AND cs_item_sk = i_item_sk
                    AND cs_sold_date_sk = d_date_sk
                    AND d_year = 1999
                    AND d_moy = 3
                    AND cs_bill_addr_sk = ca_address_sk
                    AND ca_gmt_offset = -5
             GROUP  BY i_manufact_id),
         ws
         AS (SELECT i_manufact_id,
                    Sum(ws_ext_sales_price) total_sales
             FROM   web_sales,
                    date_dim,
                    customer_address,
                    item
             WHERE  i_manufact_id IN (SELECT i_manufact_id
                                      FROM   item
                                      WHERE  i_category IN ( 'Books' ))
                    AND ws_item_sk = i_item_sk
                    AND ws_sold_date_sk = d_date_sk
                    AND d_year = 1999
                    AND d_moy = 3
                    AND ws_bill_addr_sk = ca_address_sk
                    AND ca_gmt_offset = -5
             GROUP  BY i_manufact_id)
    SELECT i_manufact_id,
                   Sum(total_sales) total_sales
    FROM   (SELECT *
            FROM   ss
            UNION ALL
            SELECT *
            FROM   cs
            UNION ALL
            SELECT *
            FROM   ws) tmp1
    GROUP  BY i_manufact_id
    ORDER  BY total_sales
    LIMIT 100;
    """


def channel_total(  # noqa: D103
    sales: pl.LazyFrame,
    date_dim: pl.LazyFrame,
    customer_address: pl.LazyFrame,
    item: pl.LazyFrame,
    books_manufacturers: pl.LazyFrame,
    *,
    sold_date_key: str,
    addr_key: str,
    item_key: str,
    price_col: str,
) -> pl.LazyFrame:
    return (
        sales.join(date_dim, left_on=sold_date_key, right_on="d_date_sk")
        .join(customer_address, left_on=addr_key, right_on="ca_address_sk")
        .join(item, left_on=item_key, right_on="i_item_sk")
        .join(books_manufacturers, on="i_manufact_id")
        .filter(
            (pl.col("d_year") == 1999)
            & (pl.col("d_moy") == 3)
            & (pl.col("ca_gmt_offset") == -5)
        )
        .group_by("i_manufact_id")
        .agg([pl.col(price_col).sum().alias("total_sales")])
    )


def polars_impl(run_config: RunConfig) -> pl.LazyFrame:
    """Query 33."""
    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    catalog_sales = get_data(
        run_config.dataset_path, "catalog_sales", run_config.suffix
    )
    web_sales = get_data(run_config.dataset_path, "web_sales", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    item = get_data(run_config.dataset_path, "item", run_config.suffix)
    customer_address = get_data(
        run_config.dataset_path, "customer_address", run_config.suffix
    )

    books_manufacturers = (
        item.filter(pl.col("i_category") == "Books").select("i_manufact_id").unique()
    )

    ss = channel_total(
        store_sales,
        date_dim,
        customer_address,
        item,
        books_manufacturers,
        sold_date_key="ss_sold_date_sk",
        addr_key="ss_addr_sk",
        item_key="ss_item_sk",
        price_col="ss_ext_sales_price",
    )
    cs = channel_total(
        catalog_sales,
        date_dim,
        customer_address,
        item,
        books_manufacturers,
        sold_date_key="cs_sold_date_sk",
        addr_key="cs_bill_addr_sk",
        item_key="cs_item_sk",
        price_col="cs_ext_sales_price",
    )
    ws = channel_total(
        web_sales,
        date_dim,
        customer_address,
        item,
        books_manufacturers,
        sold_date_key="ws_sold_date_sk",
        addr_key="ws_bill_addr_sk",
        item_key="ws_item_sk",
        price_col="ws_ext_sales_price",
    )

    return (
        pl.concat([ss, cs, ws])
        .group_by("i_manufact_id")
        .agg([pl.col("total_sales").sum().alias("total_sales")])
        .select(["i_manufact_id", "total_sales"])
        .sort("total_sales")
        .limit(100)
    )
