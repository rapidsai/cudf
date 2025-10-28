# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 35."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.utils import get_data

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 35."""
    return """
    SELECT ca_state,
                   cd_gender,
                   cd_marital_status,
                   cd_dep_count,
                   Count(*) cnt1,
                   Stddev_samp(cd_dep_count),
                   Avg(cd_dep_count),
                   Max(cd_dep_count),
                   cd_dep_employed_count,
                   Count(*) cnt2,
                   Stddev_samp(cd_dep_employed_count),
                   Avg(cd_dep_employed_count),
                   Max(cd_dep_employed_count),
                   cd_dep_college_count,
                   Count(*) cnt3,
                   Stddev_samp(cd_dep_college_count),
                   Avg(cd_dep_college_count),
                   Max(cd_dep_college_count)
    FROM   customer c,
           customer_address ca,
           customer_demographics
    WHERE  c.c_current_addr_sk = ca.ca_address_sk
           AND cd_demo_sk = c.c_current_cdemo_sk
           AND EXISTS (SELECT *
                       FROM   store_sales,
                              date_dim
                       WHERE  c.c_customer_sk = ss_customer_sk
                              AND ss_sold_date_sk = d_date_sk
                              AND d_year = 2001
                              AND d_qoy < 4)
           AND ( EXISTS (SELECT *
                         FROM   web_sales,
                                date_dim
                         WHERE  c.c_customer_sk = ws_bill_customer_sk
                                AND ws_sold_date_sk = d_date_sk
                                AND d_year = 2001
                                AND d_qoy < 4)
                  OR EXISTS (SELECT *
                             FROM   catalog_sales,
                                    date_dim
                             WHERE  c.c_customer_sk = cs_ship_customer_sk
                                    AND cs_sold_date_sk = d_date_sk
                                    AND d_year = 2001
                                    AND d_qoy < 4) )
    GROUP  BY ca_state,
              cd_gender,
              cd_marital_status,
              cd_dep_count,
              cd_dep_employed_count,
              cd_dep_college_count
    ORDER  BY ca_state,
              cd_gender,
              cd_marital_status,
              cd_dep_count,
              cd_dep_employed_count,
              cd_dep_college_count
    LIMIT 100;
    """


def polars_impl(run_config: RunConfig) -> pl.LazyFrame:
    """Query 35."""
    # Load tables
    customer = get_data(run_config.dataset_path, "customer", run_config.suffix)
    customer_address = get_data(
        run_config.dataset_path, "customer_address", run_config.suffix
    )
    customer_demographics = get_data(
        run_config.dataset_path, "customer_demographics", run_config.suffix
    )
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    web_sales = get_data(run_config.dataset_path, "web_sales", run_config.suffix)
    catalog_sales = get_data(
        run_config.dataset_path, "catalog_sales", run_config.suffix
    )
    # Main customer data
    main_customers = customer.join(
        customer_address, left_on="c_current_addr_sk", right_on="ca_address_sk"
    ).join(customer_demographics, left_on="c_current_cdemo_sk", right_on="cd_demo_sk")
    # Store sales existence check
    ss_customers = (
        store_sales.join(date_dim, left_on="ss_sold_date_sk", right_on="d_date_sk")
        .filter((pl.col("d_year") == 2001) & (pl.col("d_qoy") < 4))
        .select("ss_customer_sk")
        .unique()
    )
    # Web sales existence check
    ws_customers = (
        web_sales.join(date_dim, left_on="ws_sold_date_sk", right_on="d_date_sk")
        .filter((pl.col("d_year") == 2001) & (pl.col("d_qoy") < 4))
        .select("ws_bill_customer_sk")
        .unique()
    )
    # Catalog sales existence check
    cs_customers = (
        catalog_sales.join(date_dim, left_on="cs_sold_date_sk", right_on="d_date_sk")
        .filter((pl.col("d_year") == 2001) & (pl.col("d_qoy") < 4))
        .select("cs_ship_customer_sk")
        .unique()
    )
    # Apply EXISTS conditions
    return (
        main_customers.join(
            ss_customers, left_on="c_customer_sk", right_on="ss_customer_sk"
        )
        .join(
            pl.concat(
                [
                    ws_customers.rename({"ws_bill_customer_sk": "customer_sk"}),
                    cs_customers.rename({"cs_ship_customer_sk": "customer_sk"}),
                ]
            ).unique(),
            left_on="c_customer_sk",
            right_on="customer_sk",
        )
        .group_by(
            [
                "ca_state",
                "cd_gender",
                "cd_marital_status",
                "cd_dep_count",
                "cd_dep_employed_count",
                "cd_dep_college_count",
            ]
        )
        .agg(
            [
                # Cast -> Int64 to match DuckDB
                pl.len().cast(pl.Int64).alias("cnt1"),
                pl.col("cd_dep_count").std().alias("stddev_samp(cd_dep_count)"),
                pl.col("cd_dep_count").mean().alias("avg(cd_dep_count)"),
                pl.col("cd_dep_count").max().alias("max(cd_dep_count)"),
                # Cast -> Int64 to match DuckDB
                pl.len().cast(pl.Int64).alias("cnt2"),
                pl.col("cd_dep_employed_count")
                .std()
                .alias("stddev_samp(cd_dep_employed_count)"),
                pl.col("cd_dep_employed_count")
                .mean()
                .alias("avg(cd_dep_employed_count)"),
                pl.col("cd_dep_employed_count")
                .max()
                .alias("max(cd_dep_employed_count)"),
                # Cast -> Int64 to match DuckDB
                pl.len().cast(pl.Int64).alias("cnt3"),
                pl.col("cd_dep_college_count")
                .std()
                .alias("stddev_samp(cd_dep_college_count)"),
                pl.col("cd_dep_college_count")
                .mean()
                .alias("avg(cd_dep_college_count)"),
                pl.col("cd_dep_college_count").max().alias("max(cd_dep_college_count)"),
            ]
        )
        .select(
            [
                "ca_state",
                "cd_gender",
                "cd_marital_status",
                "cd_dep_count",
                "cnt1",
                "stddev_samp(cd_dep_count)",
                "avg(cd_dep_count)",
                "max(cd_dep_count)",
                "cd_dep_employed_count",
                "cnt2",
                "stddev_samp(cd_dep_employed_count)",
                "avg(cd_dep_employed_count)",
                "max(cd_dep_employed_count)",
                "cd_dep_college_count",
                "cnt3",
                "stddev_samp(cd_dep_college_count)",
                "avg(cd_dep_college_count)",
                "max(cd_dep_college_count)",
            ]
        )
        .sort(
            [
                "ca_state",
                "cd_gender",
                "cd_marital_status",
                "cd_dep_count",
                "cd_dep_employed_count",
                "cd_dep_college_count",
            ],
            nulls_last=True,
        )
        .limit(100)
    )
