# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 16."""

from __future__ import annotations

from datetime import date, timedelta
from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.utils import get_data

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 16."""
    return """
    SELECT
             Count(DISTINCT cs_order_number) AS 'order count' ,
             Sum(cs_ext_ship_cost)           AS 'total shipping cost' ,
             Sum(cs_net_profit)              AS 'total net profit'
    FROM     catalog_sales cs1 ,
             date_dim ,
             customer_address ,
             call_center
    WHERE    d_date BETWEEN '2002-3-01' AND      (
                      Cast('2002-3-01' AS DATE) + INTERVAL '60' day)
    AND      cs1.cs_ship_date_sk = d_date_sk
    AND      cs1.cs_ship_addr_sk = ca_address_sk
    AND      ca_state = 'IA'
    AND      cs1.cs_call_center_sk = cc_call_center_sk
    AND      cc_county IN ('Williamson County',
                           'Williamson County',
                           'Williamson County',
                           'Williamson County',
                           'Williamson County' )
    AND      EXISTS
             (
                    SELECT *
                    FROM   catalog_sales cs2
                    WHERE  cs1.cs_order_number = cs2.cs_order_number
                    AND    cs1.cs_warehouse_sk <> cs2.cs_warehouse_sk)
    AND      NOT EXISTS
             (
                    SELECT *
                    FROM   catalog_returns cr1
                    WHERE  cs1.cs_order_number = cr1.cr_order_number)
    ORDER BY count(DISTINCT cs_order_number)
    LIMIT 100;
    """


def polars_impl(run_config: RunConfig) -> pl.LazyFrame:
    """Query 16."""
    # Load tables
    catalog_sales = get_data(
        run_config.dataset_path, "catalog_sales", run_config.suffix
    )
    catalog_returns = get_data(
        run_config.dataset_path, "catalog_returns", run_config.suffix
    )
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    customer_address = get_data(
        run_config.dataset_path, "customer_address", run_config.suffix
    )
    call_center = get_data(run_config.dataset_path, "call_center", run_config.suffix)
    start_date = date(2002, 3, 1)
    end_date = start_date + timedelta(days=60)
    # First apply basic filters to catalog_sales
    filtered_sales = (
        catalog_sales.join(date_dim, left_on="cs_ship_date_sk", right_on="d_date_sk")
        .join(customer_address, left_on="cs_ship_addr_sk", right_on="ca_address_sk")
        .join(call_center, left_on="cs_call_center_sk", right_on="cc_call_center_sk")
        .filter(
            pl.col("d_date").is_between(start_date, end_date, closed="both")
            & (pl.col("ca_state") == "IA")
            & pl.col("cc_county").is_in(["Williamson County"])
        )
    )
    # Handle EXISTS condition: for each row, check if there's another row with same order but different warehouse
    # Self-join to find rows that have other rows with same order number but different warehouse
    exists_condition = (
        filtered_sales.with_row_index("row_id")
        .join(
            catalog_sales.select(["cs_order_number", "cs_warehouse_sk"]).rename(
                {"cs_warehouse_sk": "other_warehouse_sk"}
            ),
            on="cs_order_number",
        )
        .filter(pl.col("cs_warehouse_sk") != pl.col("other_warehouse_sk"))
        .select("row_id")
        .unique()
    )
    # Handle NOT EXISTS condition: orders that don't have returns
    returned_orders = catalog_returns.select("cr_order_number").unique()
    return (
        filtered_sales.with_row_index("row_id")
        .join(exists_condition, on="row_id")
        .join(
            returned_orders,
            left_on="cs_order_number",
            right_on="cr_order_number",
            how="anti",
        )
        .select(
            [
                # Cast -> Int64 to match DuckDB
                pl.col("cs_order_number")
                .n_unique()
                .cast(pl.Int64)
                .alias("order count"),
                pl.col("cs_ext_ship_cost").sum().alias("total shipping cost"),
                pl.col("cs_net_profit").sum().alias("total net profit"),
            ]
        )
        .sort(["order count"])
        .limit(100)
    )
