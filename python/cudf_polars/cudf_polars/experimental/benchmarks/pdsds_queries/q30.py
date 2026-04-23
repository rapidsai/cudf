# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 30."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.pdsds_parameters import load_parameters
from cudf_polars.experimental.benchmarks.utils import QueryResult, get_data

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 30."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=30,
        qualification=run_config.qualification,
    )

    year = params["year"]
    state = params["state"]

    return f"""
    WITH customer_total_return
         AS (SELECT wr_returning_customer_sk AS ctr_customer_sk,
                    ca_state                 AS ctr_state,
                    Sum(wr_return_amt)       AS ctr_total_return
             FROM   web_returns,
                    date_dim,
                    customer_address
             WHERE  wr_returned_date_sk = d_date_sk
                    AND d_year = {year}
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
           AND ca_state = '{state}'
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


def polars_impl(run_config: RunConfig) -> QueryResult:
    """Query 30."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=30,
        qualification=run_config.qualification,
    )

    year = params["year"]
    state = params["state"]

    # Load tables
    web_returns = get_data(run_config.dataset_path, "web_returns", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    customer_address = get_data(
        run_config.dataset_path, "customer_address", run_config.suffix
    )
    customer = get_data(run_config.dataset_path, "customer", run_config.suffix)
    # CTE: customer_total_return
    customer_total_return = (
        web_returns.join(date_dim, how="cross")
        .join(customer_address, how="cross")
        .filter(
            (pl.col("wr_returned_date_sk") == pl.col("d_date_sk"))
            & (pl.col("d_year") == year)
            & (pl.col("wr_returning_addr_sk") == pl.col("ca_address_sk"))
        )
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
    sort_by = {
        "c_customer_id": False,
        "c_salutation": False,
        "c_first_name": False,
        "c_last_name": False,
        "c_preferred_cust_flag": False,
        "c_birth_day": False,
        "c_birth_month": False,
        "c_birth_year": False,
        "c_birth_country": False,
        "c_login": False,
        "c_email_address": False,
        "c_last_review_date_sk": False,
        "ctr_total_return": False,
    }
    limit = 100
    return QueryResult(
        frame=(
            customer.join(
                customer_address, left_on="c_current_addr_sk", right_on="ca_address_sk"
            )
            .filter(pl.col("ca_state") == state)
            .join(
                qualified_customers, left_on="c_customer_sk", right_on="ctr_customer_sk"
            )
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
            .sort(sort_by.keys(), nulls_last=True)
            .limit(limit)
        ),
        sort_by=list(sort_by.items()),
        limit=limit,
    )


def polars_impl_naive(run_config: RunConfig) -> QueryResult:
    """Query 30 (naive)."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=30,
        qualification=run_config.qualification,
    )

    year = params["year"]
    state = params["state"]

    web_returns = get_data(run_config.dataset_path, "web_returns", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    customer_address = get_data(
        run_config.dataset_path, "customer_address", run_config.suffix
    )
    customer = get_data(run_config.dataset_path, "customer", run_config.suffix)

    # SQL: CTE customer_total_return — FROM web_returns, date_dim, customer_address WHERE wr_returned_date_sk=d_date_sk AND d_year={year} AND wr_returning_addr_sk=ca_address_sk GROUP BY wr_returning_customer_sk, ca_state
    customer_total_return = (
        # SQL: CROSS JOIN date_dim, customer_address (filter applied below)
        web_returns.join(date_dim, how="cross")
        .join(customer_address, how="cross")
        # SQL: WHERE wr_returned_date_sk=d_date_sk AND d_year={year} AND wr_returning_addr_sk=ca_address_sk
        .filter(
            (pl.col("wr_returned_date_sk") == pl.col("d_date_sk"))
            & (pl.col("d_year") == year)
            & (pl.col("wr_returning_addr_sk") == pl.col("ca_address_sk"))
        )
        # SQL: GROUP BY wr_returning_customer_sk AS ctr_customer_sk, ca_state AS ctr_state
        .group_by(
            [
                pl.col("wr_returning_customer_sk").alias("ctr_customer_sk"),
                pl.col("ca_state").alias("ctr_state"),
            ]
        )
        # SQL: Sum(wr_return_amt) AS ctr_total_return
        .agg([pl.col("wr_return_amt").sum().alias("ctr_total_return")])
    )

    # Pre-computed correlated subquery — structurally required for Polars
    # SQL: correlated subquery — Avg(ctr_total_return)*1.2 per state
    state_averages = customer_total_return.group_by("ctr_state").agg(
        [(pl.col("ctr_total_return").mean() * 1.2).alias("avg_return")]
    )

    # SQL: WHERE ctr1.ctr_total_return > Avg(ctr2.ctr_total_return)*1.2 WHERE ctr1.ctr_state=ctr2.ctr_state
    qualified_customers = customer_total_return.join(
        state_averages, on="ctr_state"
    ).filter(pl.col("ctr_total_return") > pl.col("avg_return"))

    return QueryResult(
        frame=(
            # SQL: FROM customer_total_return ctr1, customer_address, customer
            qualified_customers.join(customer_address, how="cross")
            # SQL: CROSS JOIN customer, customer_address (filter applied below)
            .join(customer, how="cross")
            # SQL: WHERE ca_address_sk=c_current_addr_sk AND ca_state='{state}' AND ctr_customer_sk=c_customer_sk AND ctr_total_return>avg*1.2
            .filter(
                (pl.col("ca_address_sk") == pl.col("c_current_addr_sk"))
                & (pl.col("ca_state") == state)
                & (pl.col("ctr_customer_sk") == pl.col("c_customer_sk"))
            )
            # SQL: SELECT c_customer_id ... ctr_total_return
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
            # SQL: ORDER BY c_customer_id, c_salutation, c_first_name, ... ctr_total_return
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
                ],
                nulls_last=True,
            )
            # SQL: LIMIT 100
            .limit(100)
        ),
        sort_by=[
            ("c_customer_id", False),
            ("c_salutation", False),
            ("c_first_name", False),
            ("c_last_name", False),
            ("c_preferred_cust_flag", False),
            ("c_birth_day", False),
            ("c_birth_month", False),
            ("c_birth_year", False),
            ("c_birth_country", False),
            ("c_login", False),
            ("c_email_address", False),
            ("c_last_review_date_sk", False),
            ("ctr_total_return", False),
        ],
        limit=100,
    )
