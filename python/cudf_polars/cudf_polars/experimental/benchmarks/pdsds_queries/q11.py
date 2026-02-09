# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 11."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.utils import get_data

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig

# Target years
YEAR_FIRST = 2001
YEAR_SECOND = 2002


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 11."""
    return f"""
    WITH year_total
         AS (
           SELECT c_customer_id                                customer_id,
                  c_first_name                                 customer_first_name,
                  c_last_name                                  customer_last_name,
                  c_preferred_cust_flag                        customer_preferred_cust_flag,
                  c_birth_country                              customer_birth_country,
                  c_login                                      customer_login,
                  c_email_address                              customer_email_address,
                  d_year                                       dyear,
                  SUM(ss_ext_list_price - ss_ext_discount_amt) year_total,
                  's'                                          sale_type
           FROM   customer,
                  store_sales,
                  date_dim
           WHERE  c_customer_sk = ss_customer_sk
                  AND ss_sold_date_sk = d_date_sk
           GROUP  BY c_customer_id,
                     c_first_name,
                     c_last_name,
                     c_preferred_cust_flag,
                     c_birth_country,
                     c_login,
                     c_email_address,
                     d_year
           UNION ALL
           SELECT c_customer_id                                customer_id,
                  c_first_name                                 customer_first_name,
                  c_last_name                                  customer_last_name,
                  c_preferred_cust_flag                        customer_preferred_cust_flag,
                  c_birth_country                              customer_birth_country,
                  c_login                                      customer_login,
                  c_email_address                              customer_email_address,
                  d_year                                       dyear,
                  SUM(ws_ext_list_price - ws_ext_discount_amt) year_total,
                  'w'                                          sale_type
           FROM   customer,
                  web_sales,
                  date_dim
           WHERE  c_customer_sk = ws_bill_customer_sk
                  AND ws_sold_date_sk = d_date_sk
           GROUP  BY c_customer_id,
                     c_first_name,
                     c_last_name,
                     c_preferred_cust_flag,
                     c_birth_country,
                     c_login,
                     c_email_address,
                     d_year
         )
    SELECT t_s_secyear.customer_id,
           t_s_secyear.customer_first_name,
           t_s_secyear.customer_last_name,
           t_s_secyear.customer_birth_country
    FROM   year_total t_s_firstyear,
           year_total t_s_secyear,
           year_total t_w_firstyear,
           year_total t_w_secyear
    WHERE  t_s_secyear.customer_id = t_s_firstyear.customer_id
           AND t_s_firstyear.customer_id = t_w_secyear.customer_id
           AND t_s_firstyear.customer_id = t_w_firstyear.customer_id
           AND t_s_firstyear.sale_type = 's'
           AND t_w_firstyear.sale_type = 'w'
           AND t_s_secyear.sale_type = 's'
           AND t_w_secyear.sale_type = 'w'
           AND t_s_firstyear.dyear = {YEAR_FIRST}
           AND t_s_secyear.dyear = {YEAR_SECOND}
           AND t_w_firstyear.dyear = {YEAR_FIRST}
           AND t_w_secyear.dyear = {YEAR_SECOND}
           AND t_s_firstyear.year_total > 0
           AND t_w_firstyear.year_total > 0
           AND CASE
                 WHEN t_w_firstyear.year_total > 0 THEN
                   t_w_secyear.year_total / t_w_firstyear.year_total
                 ELSE 0.0
               END
             > CASE
                 WHEN t_s_firstyear.year_total > 0 THEN
                   t_s_secyear.year_total / t_s_firstyear.year_total
                 ELSE 0.0
               END
    ORDER  BY t_s_secyear.customer_id,
              t_s_secyear.customer_first_name,
              t_s_secyear.customer_last_name,
              t_s_secyear.customer_birth_country
    LIMIT 100;
    """


def create_year_total(
    customer_table: pl.LazyFrame,
    sales_table: pl.LazyFrame,
    customer_sk_col: str,
    date_sk_col: str,
    list_price_col: str,
    discount_col: str,
    year_filter: pl.LazyFrame,
) -> pl.LazyFrame:
    """Computes per-customer yearly totals for a sales table and year."""
    return (
        customer_table.join(
            sales_table, left_on="c_customer_sk", right_on=customer_sk_col, how="inner"
        )
        .join(year_filter, left_on=date_sk_col, right_on="d_date_sk", how="inner")
        .group_by(
            [
                "c_customer_id",
                "c_first_name",
                "c_last_name",
                "c_preferred_cust_flag",
                "c_birth_country",
                "c_login",
                "c_email_address",
                "d_year",
            ]
        )
        .agg(
            [(pl.col(list_price_col) - pl.col(discount_col)).sum().alias("year_total")]
        )
        .select(
            [
                pl.col("c_customer_id").alias("customer_id"),
                pl.col("c_first_name").alias("customer_first_name"),
                pl.col("c_last_name").alias("customer_last_name"),
                pl.col("c_preferred_cust_flag").alias("customer_preferred_cust_flag"),
                pl.col("c_birth_country").alias("customer_birth_country"),
                pl.col("c_login").alias("customer_login"),
                pl.col("c_email_address").alias("customer_email_address"),
                pl.col("d_year").alias("dyear"),
                pl.col("year_total"),
            ]
        )
    )


def polars_impl(run_config: RunConfig) -> pl.LazyFrame:
    """Query 11."""
    # Load required tables
    customer = get_data(run_config.dataset_path, "customer", run_config.suffix)
    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    web_sales = get_data(run_config.dataset_path, "web_sales", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)

    date_first = date_dim.filter(pl.col("d_year") == YEAR_FIRST)
    date_second = date_dim.filter(pl.col("d_year") == YEAR_SECOND)

    # Total store sales 2001
    t_s_firstyear = create_year_total(
        customer,
        store_sales,
        "ss_customer_sk",
        "ss_sold_date_sk",
        "ss_ext_list_price",
        "ss_ext_discount_amt",
        date_first,
    ).select(
        [
            pl.col("customer_id").alias("s_first_customer_id"),
            pl.col("year_total").alias("s_first_year_total"),
        ]
    )

    # Total store sales 2002
    t_s_secyear = create_year_total(
        customer,
        store_sales,
        "ss_customer_sk",
        "ss_sold_date_sk",
        "ss_ext_list_price",
        "ss_ext_discount_amt",
        date_second,
    ).select(
        [
            pl.col("customer_id").alias("s_sec_customer_id"),
            pl.col("customer_first_name"),
            pl.col("customer_last_name"),
            pl.col("customer_birth_country"),
            pl.col("year_total").alias("s_sec_year_total"),
        ]
    )

    # Total web sales 2001
    t_w_firstyear = create_year_total(
        customer,
        web_sales,
        "ws_bill_customer_sk",
        "ws_sold_date_sk",
        "ws_ext_list_price",
        "ws_ext_discount_amt",
        date_first,
    ).select(
        [
            pl.col("customer_id").alias("w_first_customer_id"),
            pl.col("year_total").alias("w_first_year_total"),
        ]
    )

    # Total web sales 2002
    t_w_secyear = create_year_total(
        customer,
        web_sales,
        "ws_bill_customer_sk",
        "ws_sold_date_sk",
        "ws_ext_list_price",
        "ws_ext_discount_amt",
        date_second,
    ).select(
        [
            pl.col("customer_id").alias("w_sec_customer_id"),
            pl.col("year_total").alias("w_sec_year_total"),
        ]
    )

    # Join the tables and filter to get customers whose web and store spending grew from 2001 -> 2002
    return (
        t_s_secyear.join(
            t_s_firstyear,
            left_on="s_sec_customer_id",
            right_on="s_first_customer_id",
            how="inner",
        )
        .join(
            t_w_firstyear,
            left_on="s_sec_customer_id",
            right_on="w_first_customer_id",
            how="inner",
        )
        .join(
            t_w_secyear,
            left_on="s_sec_customer_id",
            right_on="w_sec_customer_id",
            how="inner",
        )
        .filter(
            (pl.col("s_first_year_total") > 0)
            & (pl.col("w_first_year_total") > 0)
            & (
                pl.when(pl.col("w_first_year_total") > 0)
                .then(pl.col("w_sec_year_total") / pl.col("w_first_year_total"))
                .otherwise(0.0)
                > pl.when(pl.col("s_first_year_total") > 0)
                .then(pl.col("s_sec_year_total") / pl.col("s_first_year_total"))
                .otherwise(0.0)
            )
        )
        .select(
            [
                pl.col("s_sec_customer_id").alias("customer_id"),
                pl.col("customer_first_name"),
                pl.col("customer_last_name"),
                pl.col("customer_birth_country"),
            ]
        )
        .sort(
            [
                "customer_id",
                "customer_first_name",
                "customer_last_name",
                "customer_birth_country",
            ],
            nulls_last=True,
        )
        .limit(100)
    )
