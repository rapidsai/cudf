# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 31."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.utils import get_data

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 31."""
    return """
    WITH ss
         AS (SELECT ca_county,
                    d_qoy,
                    d_year,
                    Sum(ss_ext_sales_price) AS store_sales
             FROM   store_sales,
                    date_dim,
                    customer_address
             WHERE  ss_sold_date_sk = d_date_sk
                    AND ss_addr_sk = ca_address_sk
             GROUP  BY ca_county,
                       d_qoy,
                       d_year),
         ws
         AS (SELECT ca_county,
                    d_qoy,
                    d_year,
                    Sum(ws_ext_sales_price) AS web_sales
             FROM   web_sales,
                    date_dim,
                    customer_address
             WHERE  ws_sold_date_sk = d_date_sk
                    AND ws_bill_addr_sk = ca_address_sk
             GROUP  BY ca_county,
                       d_qoy,
                       d_year)
    SELECT ss1.ca_county,
           ss1.d_year,
           ws2.web_sales / ws1.web_sales     web_q1_q2_increase,
           ss2.store_sales / ss1.store_sales store_q1_q2_increase,
           ws3.web_sales / ws2.web_sales     web_q2_q3_increase,
           ss3.store_sales / ss2.store_sales store_q2_q3_increase
    FROM   ss ss1,
           ss ss2,
           ss ss3,
           ws ws1,
           ws ws2,
           ws ws3
    WHERE  ss1.d_qoy = 1
           AND ss1.d_year = 2001
           AND ss1.ca_county = ss2.ca_county
           AND ss2.d_qoy = 2
           AND ss2.d_year = 2001
           AND ss2.ca_county = ss3.ca_county
           AND ss3.d_qoy = 3
           AND ss3.d_year = 2001
           AND ss1.ca_county = ws1.ca_county
           AND ws1.d_qoy = 1
           AND ws1.d_year = 2001
           AND ws1.ca_county = ws2.ca_county
           AND ws2.d_qoy = 2
           AND ws2.d_year = 2001
           AND ws1.ca_county = ws3.ca_county
           AND ws3.d_qoy = 3
           AND ws3.d_year = 2001
           AND CASE
                 WHEN ws1.web_sales > 0 THEN ws2.web_sales / ws1.web_sales
                 ELSE NULL
               END > CASE
                       WHEN ss1.store_sales > 0 THEN
                       ss2.store_sales / ss1.store_sales
                       ELSE NULL
                     END
           AND CASE
                 WHEN ws2.web_sales > 0 THEN ws3.web_sales / ws2.web_sales
                 ELSE NULL
               END > CASE
                       WHEN ss2.store_sales > 0 THEN
                       ss3.store_sales / ss2.store_sales
                       ELSE NULL
                     END
    ORDER  BY ss1.ca_county,
              ss1.d_year,
              web_q1_q2_increase,
              store_q1_q2_increase,
              web_q2_q3_increase,
              store_q2_q3_increase;
    """


def polars_impl(run_config: RunConfig) -> pl.LazyFrame:
    """Query 31."""
    # Load tables
    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    web_sales = get_data(run_config.dataset_path, "web_sales", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    customer_address = get_data(
        run_config.dataset_path, "customer_address", run_config.suffix
    )

    # CTE: ss (store sales by county, quarter, year)
    ss = (
        store_sales.join(date_dim, left_on="ss_sold_date_sk", right_on="d_date_sk")
        .join(customer_address, left_on="ss_addr_sk", right_on="ca_address_sk")
        .group_by(["ca_county", "d_qoy", "d_year"])
        .agg([pl.col("ss_ext_sales_price").sum().alias("store_sales")])
    )

    # CTE: ws (web sales by county, quarter, year)
    ws = (
        web_sales.join(date_dim, left_on="ws_sold_date_sk", right_on="d_date_sk")
        .join(customer_address, left_on="ws_bill_addr_sk", right_on="ca_address_sk")
        .group_by(["ca_county", "d_qoy", "d_year"])
        .agg([pl.col("ws_ext_sales_price").sum().alias("web_sales")])
    )

    # Create filtered versions for each quarter
    ss1 = ss.filter((pl.col("d_qoy") == 1) & (pl.col("d_year") == 2001))
    ss2 = ss.filter((pl.col("d_qoy") == 2) & (pl.col("d_year") == 2001))
    ss3 = ss.filter((pl.col("d_qoy") == 3) & (pl.col("d_year") == 2001))
    ws1 = ws.filter((pl.col("d_qoy") == 1) & (pl.col("d_year") == 2001))
    ws2 = ws.filter((pl.col("d_qoy") == 2) & (pl.col("d_year") == 2001))
    ws3 = ws.filter((pl.col("d_qoy") == 3) & (pl.col("d_year") == 2001))

    # Join all quarters together by county
    return (
        ss1.join(ss2, on="ca_county", suffix="_q2")
        .join(ss3, on="ca_county", suffix="_q3")
        .join(ws1, on="ca_county", suffix="_ws1")
        .join(ws2, on="ca_county", suffix="_ws2")
        .join(ws3, on="ca_county", suffix="_ws3")
        .with_columns(
            [
                # Calculate ratios with null handling
                pl.when(pl.col("web_sales") > 0)
                .then(pl.col("web_sales_ws2") / pl.col("web_sales"))
                .otherwise(None)
                .alias("web_q1_q2_increase"),
                pl.when(pl.col("store_sales") > 0)
                .then(pl.col("store_sales_q2") / pl.col("store_sales"))
                .otherwise(None)
                .alias("store_q1_q2_increase"),
                pl.when(pl.col("web_sales_ws2") > 0)
                .then(pl.col("web_sales_ws3") / pl.col("web_sales_ws2"))
                .otherwise(None)
                .alias("web_q2_q3_increase"),
                pl.when(pl.col("store_sales_q2") > 0)
                .then(pl.col("store_sales_q3") / pl.col("store_sales_q2"))
                .otherwise(None)
                .alias("store_q2_q3_increase"),
            ]
        )
        .filter(
            # First condition: web_q1_q2 > store_q1_q2
            (
                pl.when(pl.col("web_sales") > 0)
                .then(pl.col("web_sales_ws2") / pl.col("web_sales"))
                .otherwise(None)
                > pl.when(pl.col("store_sales") > 0)
                .then(pl.col("store_sales_q2") / pl.col("store_sales"))
                .otherwise(None)
            )
            &
            # Second condition: web_q2_q3 > store_q2_q3
            (
                pl.when(pl.col("web_sales_ws2") > 0)
                .then(pl.col("web_sales_ws3") / pl.col("web_sales_ws2"))
                .otherwise(None)
                > pl.when(pl.col("store_sales_q2") > 0)
                .then(pl.col("store_sales_q3") / pl.col("store_sales_q2"))
                .otherwise(None)
            )
        )
        .select(
            [
                pl.col("ca_county"),
                pl.col("d_year"),
                "web_q1_q2_increase",
                "store_q1_q2_increase",
                "web_q2_q3_increase",
                "store_q2_q3_increase",
            ]
        )
        .sort(
            [
                "ca_county",
                "d_year",
                "web_q1_q2_increase",
                "store_q1_q2_increase",
                "web_q2_q3_increase",
                "store_q2_q3_increase",
            ]
        )
    )
