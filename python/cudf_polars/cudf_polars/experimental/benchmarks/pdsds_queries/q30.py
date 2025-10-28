# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 30."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.utils import get_data

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 30."""
    return """
    WITH customer_total_return
         AS (SELECT wr_returning_customer_sk AS ctr_customer_sk,
                    ca_state                 AS ctr_state,
                    Sum(wr_return_amt)       AS ctr_total_return
             FROM   web_returns,
                    date_dim,
                    customer_address
             WHERE  wr_returned_date_sk = d_date_sk
                    AND d_year = 2000
                    AND wr_returning_addr_sk = ca_address_sk
             GROUP  BY wr_returning_customer_sk,
                       ca_state)
    SELECT c_customer_id,
                   c_salutation,
                   c_first_name,
                   c_last_name,
                   c_preferred_cust_flag,
                   c_birth_day,
                   c_birth_month,
                   c_birth_year,
                   c_birth_country,
                   c_login,
                   c_email_address,
                   c_last_review_date_sk,
                   ctr_total_return
    FROM   customer_total_return ctr1,
           customer_address,
           customer
    WHERE  ctr1.ctr_total_return > (SELECT Avg(ctr_total_return) * 1.2
                                    FROM   customer_total_return ctr2
                                    WHERE  ctr1.ctr_state = ctr2.ctr_state)
           AND ca_address_sk = c_current_addr_sk
           AND ca_state = 'IN'
           AND ctr1.ctr_customer_sk = c_customer_sk
    ORDER  BY c_customer_id,
              c_salutation,
              c_first_name,
              c_last_name,
              c_preferred_cust_flag,
              c_birth_day,
              c_birth_month,
              c_birth_year,
              c_birth_country,
              c_login,
              c_email_address,
              c_last_review_date_sk,
              ctr_total_return
    LIMIT 100;
    """


def polars_impl(run_config: RunConfig) -> pl.LazyFrame:
    """Query 30."""
    # Load tables
    web_returns = get_data(run_config.dataset_path, "web_returns", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    customer_address = get_data(
        run_config.dataset_path, "customer_address", run_config.suffix
    )
    customer = get_data(run_config.dataset_path, "customer", run_config.suffix)
    # CTE: customer_total_return
    customer_total_return = (
        web_returns.join(date_dim, left_on="wr_returned_date_sk", right_on="d_date_sk")
        .join(
            customer_address, left_on="wr_returning_addr_sk", right_on="ca_address_sk"
        )
        .filter(pl.col("d_year") == 2000)
        .group_by(
            [
                pl.col("wr_returning_customer_sk").alias("ctr_customer_sk"),
                pl.col("ca_state").alias("ctr_state"),
            ]
        )
        .agg([pl.col("wr_return_amt").sum().alias("ctr_total_return")])
    )
    # Calculate state averages for the correlated subquery
    state_averages = (
        customer_total_return.group_by("ctr_state")
        .agg([pl.col("ctr_total_return").mean().alias("avg_return")])
        .with_columns([pl.col("avg_return") * 1.2])
    )
    # Join customer_total_return with state averages to implement correlated subquery
    qualified_customers = customer_total_return.join(
        state_averages, left_on="ctr_state", right_on="ctr_state"
    ).filter(pl.col("ctr_total_return") > pl.col("avg_return"))
    # Join customer with customer_address first (avoiding cartesian product)
    # Then join with qualified customers
    return (
        customer.join(
            customer_address, left_on="c_current_addr_sk", right_on="ca_address_sk"
        )
        .filter(pl.col("ca_state") == "IN")
        .join(qualified_customers, left_on="c_customer_sk", right_on="ctr_customer_sk")
        .select(
            [
                "c_customer_id",
                "c_salutation",
                "c_first_name",
                "c_last_name",
                "c_preferred_cust_flag",
                "c_birth_day",
                "c_birth_month",
                "c_birth_year",
                "c_birth_country",
                "c_login",
                "c_email_address",
                "c_last_review_date_sk",
                "ctr_total_return",
            ]
        )
        .sort(
            [
                "c_customer_id",
                "c_salutation",
                "c_first_name",
                "c_last_name",
                "c_preferred_cust_flag",
                "c_birth_day",
                "c_birth_month",
                "c_birth_year",
                "c_birth_country",
                "c_login",
                "c_email_address",
                "c_last_review_date_sk",
                "ctr_total_return",
            ]
        )
        .limit(100)
    )
