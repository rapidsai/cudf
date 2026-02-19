# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 29."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.pdsds_parameters import load_parameters
from cudf_polars.experimental.benchmarks.utils import get_data

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 29."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=29,
        qualification=run_config.qualification,
    )

    year = params["year"]
    month = params["month"]
    agg = params["agg"]

    return f"""
    SELECT i_item_id,
                   i_item_desc,
                   s_store_id,
                   s_store_name,
                   {agg}(ss_quantity)        AS store_sales_quantity,
                   {agg}(sr_return_quantity) AS store_returns_quantity,
                   {agg}(cs_quantity)        AS catalog_sales_quantity
    FROM   store_sales,
           store_returns,
           catalog_sales,
           date_dim d1,
           date_dim d2,
           date_dim d3,
           store,
           item
    WHERE  d1.d_moy = {month}
           AND d1.d_year = {year}
           AND d1.d_date_sk = ss_sold_date_sk
           AND i_item_sk = ss_item_sk
           AND s_store_sk = ss_store_sk
           AND ss_customer_sk = sr_customer_sk
           AND ss_item_sk = sr_item_sk
           AND ss_ticket_number = sr_ticket_number
           AND sr_returned_date_sk = d2.d_date_sk
           AND d2.d_moy BETWEEN {month} AND {month} + 3
           AND d2.d_year = {year}
           AND sr_customer_sk = cs_bill_customer_sk
           AND sr_item_sk = cs_item_sk
           AND cs_sold_date_sk = d3.d_date_sk
           AND d3.d_year IN ( {year}, {year} + 1, {year} + 2 )
    GROUP  BY i_item_id,
              i_item_desc,
              s_store_id,
              s_store_name
    ORDER  BY i_item_id,
              i_item_desc,
              s_store_id,
              s_store_name
    LIMIT 100;
    """


def polars_impl(run_config: RunConfig) -> pl.LazyFrame:
    """Query 29."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=29,
        qualification=run_config.qualification,
    )

    year = params["year"]
    month = params["month"]
    agg = params["agg"]

    # Map SQL aggregation functions to Polars method names
    polars_agg = "mean" if agg == "avg" else "std" if agg == "stddev_samp" else agg

    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    store_returns = get_data(
        run_config.dataset_path, "store_returns", run_config.suffix
    )
    catalog_sales = get_data(
        run_config.dataset_path, "catalog_sales", run_config.suffix
    )
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    store = get_data(run_config.dataset_path, "store", run_config.suffix)
    item = get_data(run_config.dataset_path, "item", run_config.suffix)

    d1, d2 = [
        date_dim.clone().select(
            [
                pl.col("d_date_sk").alias(f"{p}_date_sk"),
                pl.col("d_moy").alias(f"{p}_moy"),
                pl.col("d_year").alias(f"{p}_year"),
            ]
        )
        for p in ("d1", "d2")
    ]
    d3 = date_dim.clone().select(
        [
            pl.col("d_date_sk").alias("d3_date_sk"),
            pl.col("d_year").alias("d3_year"),
        ]
    )

    return (
        store_sales.join(d1, left_on="ss_sold_date_sk", right_on="d1_date_sk")
        .join(item, left_on="ss_item_sk", right_on="i_item_sk")
        .join(store, left_on="ss_store_sk", right_on="s_store_sk")
        .join(
            store_returns,
            left_on=["ss_customer_sk", "ss_item_sk", "ss_ticket_number"],
            right_on=["sr_customer_sk", "sr_item_sk", "sr_ticket_number"],
        )
        .join(d2, left_on="sr_returned_date_sk", right_on="d2_date_sk")
        .join(
            catalog_sales,
            left_on=["ss_customer_sk", "ss_item_sk"],
            right_on=["cs_bill_customer_sk", "cs_item_sk"],
        )
        .join(d3, left_on="cs_sold_date_sk", right_on="d3_date_sk")
        .filter(
            (pl.col("d1_moy") == month)
            & (pl.col("d1_year") == year)
            & (pl.col("d2_moy").is_between(month, month + 3))
            & (pl.col("d2_year") == year)
            & (pl.col("d3_year").is_in([year, year + 1, year + 2]))
        )
        .group_by(["i_item_id", "i_item_desc", "s_store_id", "s_store_name"])
        .agg(
            [
                getattr(pl.col("ss_quantity"), polars_agg)().alias(
                    "store_sales_quantity"
                ),
                getattr(pl.col("sr_return_quantity"), polars_agg)().alias(
                    "store_returns_quantity"
                ),
                getattr(pl.col("cs_quantity"), polars_agg)().alias(
                    "catalog_sales_quantity"
                ),
            ]
        )
        .sort(["i_item_id", "i_item_desc", "s_store_id", "s_store_name"])
        .limit(100)
    )
