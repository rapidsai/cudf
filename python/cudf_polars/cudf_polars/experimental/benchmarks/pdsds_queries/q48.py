# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 48."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.utils import get_data

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 48."""
    return """
    SELECT Sum (ss_quantity)
    FROM   store_sales,
           store,
           customer_demographics,
           customer_address,
           date_dim
    WHERE  s_store_sk = ss_store_sk
           AND ss_sold_date_sk = d_date_sk
           AND d_year = 1999
           AND ( ( cd_demo_sk = ss_cdemo_sk
                   AND cd_marital_status = 'W'
                   AND cd_education_status = 'Secondary'
                   AND ss_sales_price BETWEEN 100.00 AND 150.00 )
                  OR ( cd_demo_sk = ss_cdemo_sk
                       AND cd_marital_status = 'M'
                       AND cd_education_status = 'Advanced Degree'
                       AND ss_sales_price BETWEEN 50.00 AND 100.00 )
                  OR ( cd_demo_sk = ss_cdemo_sk
                       AND cd_marital_status = 'D'
                       AND cd_education_status = '2 yr Degree'
                       AND ss_sales_price BETWEEN 150.00 AND 200.00 ) )
           AND ( ( ss_addr_sk = ca_address_sk
                   AND ca_country = 'United States'
                   AND ca_state IN ( 'TX', 'NE', 'MO' )
                   AND ss_net_profit BETWEEN 0 AND 2000 )
                  OR ( ss_addr_sk = ca_address_sk
                       AND ca_country = 'United States'
                       AND ca_state IN ( 'CO', 'TN', 'ND' )
                       AND ss_net_profit BETWEEN 150 AND 3000 )
                  OR ( ss_addr_sk = ca_address_sk
                       AND ca_country = 'United States'
                       AND ca_state IN ( 'OK', 'PA', 'CA' )
                       AND ss_net_profit BETWEEN 50 AND 25000 ) );
    """


def polars_impl(run_config: RunConfig) -> pl.LazyFrame:
    """Query 48."""
    # Load tables
    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    store = get_data(run_config.dataset_path, "store", run_config.suffix)
    customer_demographics = get_data(
        run_config.dataset_path, "customer_demographics", run_config.suffix
    )
    customer_address = get_data(
        run_config.dataset_path, "customer_address", run_config.suffix
    )
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    return (
        store_sales
        # Join with all required tables
        .join(store, left_on="ss_store_sk", right_on="s_store_sk")
        .join(customer_demographics, left_on="ss_cdemo_sk", right_on="cd_demo_sk")
        .join(customer_address, left_on="ss_addr_sk", right_on="ca_address_sk")
        .join(date_dim, left_on="ss_sold_date_sk", right_on="d_date_sk")
        # Apply filters
        .filter(
            # Year filter
            (pl.col("d_year") == 1999)
            &
            # Complex demographics OR conditions
            (
                # Condition 1: Widowed + Secondary + Price 100-150
                (
                    (pl.col("cd_marital_status") == "W")
                    & (pl.col("cd_education_status") == "Secondary")
                    & (pl.col("ss_sales_price").is_between(100.00, 150.00))
                )
                |
                # Condition 2: Married + Advanced Degree + Price 50-100
                (
                    (pl.col("cd_marital_status") == "M")
                    & (pl.col("cd_education_status") == "Advanced Degree")
                    & (pl.col("ss_sales_price").is_between(50.00, 100.00))
                )
                |
                # Condition 3: Divorced + 2 yr Degree + Price 150-200
                (
                    (pl.col("cd_marital_status") == "D")
                    & (pl.col("cd_education_status") == "2 yr Degree")
                    & (pl.col("ss_sales_price").is_between(150.00, 200.00))
                )
            )
            &
            # Complex geography OR conditions
            (
                # Condition 1: US + TX/NE/MO + Profit 0-2000
                (
                    (pl.col("ca_country") == "United States")
                    & (pl.col("ca_state").is_in(["TX", "NE", "MO"]))
                    & (pl.col("ss_net_profit").is_between(0, 2000))
                )
                |
                # Condition 2: US + CO/TN/ND + Profit 150-3000
                (
                    (pl.col("ca_country") == "United States")
                    & (pl.col("ca_state").is_in(["CO", "TN", "ND"]))
                    & (pl.col("ss_net_profit").is_between(150, 3000))
                )
                |
                # Condition 3: US + OK/PA/CA + Profit 50-25000
                (
                    (pl.col("ca_country") == "United States")
                    & (pl.col("ca_state").is_in(["OK", "PA", "CA"]))
                    & (pl.col("ss_net_profit").is_between(50, 25000))
                )
            )
        )
        # Aggregate - sum of quantity with null-safe handling
        .select(
            [
                pl.when(pl.col("ss_quantity").count() > 0)
                .then(pl.col("ss_quantity").sum())
                .otherwise(None)
                .alias("sum(ss_quantity)")
            ]
        )
    )
