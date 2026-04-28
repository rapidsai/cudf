# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 11."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.pdsds_parameters import load_parameters
from cudf_polars.experimental.benchmarks.utils import QueryResult, get_data

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 11."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=11,
        qualification=run_config.qualification,
    )

    year_first = params["year"]
    year_second = year_first + 1

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
           AND t_s_firstyear.dyear = {year_first}
           AND t_s_secyear.dyear = {year_second}
           AND t_w_firstyear.dyear = {year_first}
           AND t_w_secyear.dyear = {year_second}
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


def build_year_total(
    sales_table: pl.LazyFrame,
    customer_sk_col: str,
    date_sk_col: str,
    list_price_col: str,
    discount_col: str,
    year_dates: pl.LazyFrame,
) -> pl.LazyFrame:
    """
    Aggregate sales per customer_sk for a single year.

    Groups by c_customer_sk only (not the wide customer display columns)
    to reduce intermediate cardinality. Customer display columns are
    joined once at the end where needed.
    """
    return (
        sales_table.join(year_dates, left_on=date_sk_col, right_on="d_date_sk")
        .group_by(customer_sk_col)
        .agg(
            [(pl.col(list_price_col) - pl.col(discount_col)).sum().alias("year_total")]
        )
    )


def polars_impl(run_config: RunConfig) -> QueryResult:
    """Query 11."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=11,
        qualification=run_config.qualification,
    )

    year_first = params["year"]
    year_second = year_first + 1

    customer = get_data(run_config.dataset_path, "customer", run_config.suffix)
    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    web_sales = get_data(run_config.dataset_path, "web_sales", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)

    date_first = date_dim.filter(pl.col("d_year") == year_first).select("d_date_sk")
    date_second = date_dim.filter(pl.col("d_year") == year_second).select("d_date_sk")

    # Aggregate by customer_sk only — no customer table join needed yet.
    t_s_first = (
        build_year_total(
            store_sales,
            "ss_customer_sk",
            "ss_sold_date_sk",
            "ss_ext_list_price",
            "ss_ext_discount_amt",
            date_first,
        )
        .rename({"ss_customer_sk": "customer_sk", "year_total": "s_first_year_total"})
        .filter(pl.col("s_first_year_total") > 0)
    )

    t_s_sec = build_year_total(
        store_sales,
        "ss_customer_sk",
        "ss_sold_date_sk",
        "ss_ext_list_price",
        "ss_ext_discount_amt",
        date_second,
    ).rename({"ss_customer_sk": "customer_sk", "year_total": "s_sec_year_total"})

    t_w_first = (
        build_year_total(
            web_sales,
            "ws_bill_customer_sk",
            "ws_sold_date_sk",
            "ws_ext_list_price",
            "ws_ext_discount_amt",
            date_first,
        )
        .rename(
            {"ws_bill_customer_sk": "customer_sk", "year_total": "w_first_year_total"}
        )
        .filter(pl.col("w_first_year_total") > 0)
    )

    t_w_sec = build_year_total(
        web_sales,
        "ws_bill_customer_sk",
        "ws_sold_date_sk",
        "ws_ext_list_price",
        "ws_ext_discount_amt",
        date_second,
    ).rename({"ws_bill_customer_sk": "customer_sk", "year_total": "w_sec_year_total"})

    # Join all four aggregates on customer_sk, then join customer once for display cols.
    return QueryResult(
        frame=(
            t_s_sec.join(t_s_first, on="customer_sk")
            .join(t_w_first, on="customer_sk")
            .join(t_w_sec, on="customer_sk")
            .filter(
                pl.col("w_sec_year_total") / pl.col("w_first_year_total")
                > pl.col("s_sec_year_total") / pl.col("s_first_year_total")
            )
            .join(customer, left_on="customer_sk", right_on="c_customer_sk")
            .select(
                [
                    pl.col("c_customer_id").alias("customer_id"),
                    pl.col("c_first_name").alias("customer_first_name"),
                    pl.col("c_last_name").alias("customer_last_name"),
                    pl.col("c_birth_country").alias("customer_birth_country"),
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
        ),
        sort_by=[
            ("customer_id", False),
            ("customer_first_name", False),
            ("customer_last_name", False),
            ("customer_birth_country", False),
        ],
        limit=100,
    )


def polars_impl_naive(run_config: RunConfig) -> QueryResult:
    """
    Query 11 (naive).

    Not refactored to use year_sales_agg(): the helper joins with the same
    customer_key on both sides, but q11 uses different key names
    (c_customer_sk vs ss_customer_sk / ws_bill_customer_sk).
    """
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=11,
        qualification=run_config.qualification,
    )

    year_first = params["year"]
    year_second = year_first + 1

    customer = get_data(run_config.dataset_path, "customer", run_config.suffix)
    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    web_sales = get_data(run_config.dataset_path, "web_sales", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)

    customer_group_cols = [
        "c_customer_id",
        "c_first_name",
        "c_last_name",
        "c_preferred_cust_flag",
        "c_birth_country",
        "c_login",
        "c_email_address",
        "d_year",
    ]
    # SQL: CTE year_total (store) — FROM customer, store_sales, date_dim
    store_total = (
        # SQL: JOIN store_sales ON c_customer_sk = ss_customer_sk
        customer.join(store_sales, left_on="c_customer_sk", right_on="ss_customer_sk")
        # SQL: JOIN date_dim ON ss_sold_date_sk = d_date_sk
        .join(date_dim, left_on="ss_sold_date_sk", right_on="d_date_sk")
        # SQL: GROUP BY c_customer_id, c_first_name, c_last_name, c_preferred_cust_flag, c_birth_country, c_login, c_email_address, d_year
        .group_by(customer_group_cols)
        .agg(
            (pl.col("ss_ext_list_price") - pl.col("ss_ext_discount_amt"))
            .sum()
            .alias("year_total")
        )
        # SQL: sale_type = 's'
        .with_columns(pl.lit("s").alias("sale_type"))
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
                pl.col("sale_type"),
            ]
        )
    )

    # SQL: CTE year_total (web) — FROM customer, web_sales, date_dim
    web_total = (
        # SQL: JOIN web_sales ON c_customer_sk = ws_bill_customer_sk
        customer.join(
            web_sales, left_on="c_customer_sk", right_on="ws_bill_customer_sk"
        )
        # SQL: JOIN date_dim ON ws_sold_date_sk = d_date_sk
        .join(date_dim, left_on="ws_sold_date_sk", right_on="d_date_sk")
        # SQL: GROUP BY c_customer_id, c_first_name, c_last_name, c_preferred_cust_flag, c_birth_country, c_login, c_email_address, d_year
        .group_by(customer_group_cols)
        .agg(
            (pl.col("ws_ext_list_price") - pl.col("ws_ext_discount_amt"))
            .sum()
            .alias("year_total")
        )
        # SQL: sale_type = 'w'
        .with_columns(pl.lit("w").alias("sale_type"))
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
                pl.col("sale_type"),
            ]
        )
    )

    # SQL: year_total = store UNION ALL web
    year_total = pl.concat([store_total, web_total])

    # SQL: t_s_firstyear = year_total
    t_s_firstyear = year_total.rename(
        {
            "customer_id": "s_fy_customer_id",
            "customer_first_name": "s_fy_customer_first_name",
            "customer_last_name": "s_fy_customer_last_name",
            "customer_preferred_cust_flag": "s_fy_customer_preferred_cust_flag",
            "customer_birth_country": "s_fy_customer_birth_country",
            "customer_login": "s_fy_customer_login",
            "customer_email_address": "s_fy_customer_email_address",
            "dyear": "s_fy_dyear",
            "year_total": "s_fy_year_total",
            "sale_type": "s_fy_sale_type",
        }
    )
    # SQL: t_s_secyear = year_total
    t_s_secyear = year_total.rename(
        {
            "customer_id": "s_sy_customer_id",
            "customer_first_name": "s_sy_customer_first_name",
            "customer_last_name": "s_sy_customer_last_name",
            "customer_preferred_cust_flag": "s_sy_customer_preferred_cust_flag",
            "customer_birth_country": "s_sy_customer_birth_country",
            "customer_login": "s_sy_customer_login",
            "customer_email_address": "s_sy_customer_email_address",
            "dyear": "s_sy_dyear",
            "year_total": "s_sy_year_total",
            "sale_type": "s_sy_sale_type",
        }
    )
    # SQL: t_w_firstyear = year_total
    t_w_firstyear = year_total.rename(
        {
            "customer_id": "w_fy_customer_id",
            "customer_first_name": "w_fy_customer_first_name",
            "customer_last_name": "w_fy_customer_last_name",
            "customer_preferred_cust_flag": "w_fy_customer_preferred_cust_flag",
            "customer_birth_country": "w_fy_customer_birth_country",
            "customer_login": "w_fy_customer_login",
            "customer_email_address": "w_fy_customer_email_address",
            "dyear": "w_fy_dyear",
            "year_total": "w_fy_year_total",
            "sale_type": "w_fy_sale_type",
        }
    )
    # SQL: t_w_secyear = year_total
    t_w_secyear = year_total.rename(
        {
            "customer_id": "w_sy_customer_id",
            "customer_first_name": "w_sy_customer_first_name",
            "customer_last_name": "w_sy_customer_last_name",
            "customer_preferred_cust_flag": "w_sy_customer_preferred_cust_flag",
            "customer_birth_country": "w_sy_customer_birth_country",
            "customer_login": "w_sy_customer_login",
            "customer_email_address": "w_sy_customer_email_address",
            "dyear": "w_sy_dyear",
            "year_total": "w_sy_year_total",
            "sale_type": "w_sy_sale_type",
        }
    )

    # SQL: FROM year_total t_s_firstyear, year_total t_s_secyear, year_total t_w_firstyear, year_total t_w_secyear
    joined = (
        # SQL: JOIN t_s_secyear ON t_s_secyear.customer_id = t_s_firstyear.customer_id
        t_s_firstyear.join(
            t_s_secyear,
            left_on="s_fy_customer_id",
            right_on="s_sy_customer_id",
            how="inner",
        )
        # SQL: JOIN t_w_firstyear ON t_s_firstyear.customer_id = t_w_firstyear.customer_id
        .join(
            t_w_firstyear,
            left_on="s_fy_customer_id",
            right_on="w_fy_customer_id",
            how="inner",
        )
        # SQL: JOIN t_w_secyear ON t_s_firstyear.customer_id = t_w_secyear.customer_id
        .join(
            t_w_secyear,
            left_on="s_fy_customer_id",
            right_on="w_sy_customer_id",
            how="inner",
        )
        # SQL: WHERE sale_type/year filters and ratios
        .filter(
            (pl.col("s_fy_sale_type") == "s")
            & (pl.col("w_fy_sale_type") == "w")
            & (pl.col("s_sy_sale_type") == "s")
            & (pl.col("w_sy_sale_type") == "w")
            & (pl.col("s_fy_dyear") == year_first)
            & (pl.col("s_sy_dyear") == year_second)
            & (pl.col("w_fy_dyear") == year_first)
            & (pl.col("w_sy_dyear") == year_second)
            & (pl.col("s_fy_year_total") > 0)
            & (pl.col("w_fy_year_total") > 0)
            & (
                pl.when(pl.col("w_fy_year_total") > 0)
                .then(pl.col("w_sy_year_total") / pl.col("w_fy_year_total"))
                .otherwise(0.0)
                > pl.when(pl.col("s_fy_year_total") > 0)
                .then(pl.col("s_sy_year_total") / pl.col("s_fy_year_total"))
                .otherwise(0.0)
            )
        )
    )

    sort_cols = [
        "customer_id",
        "customer_first_name",
        "customer_last_name",
        "customer_birth_country",
    ]
    return QueryResult(
        frame=(
            # SQL: SELECT customer_id, customer_first_name, customer_last_name, customer_birth_country
            joined.select(
                [
                    pl.col("s_fy_customer_id").alias("customer_id"),
                    pl.col("s_sy_customer_first_name").alias("customer_first_name"),
                    pl.col("s_sy_customer_last_name").alias("customer_last_name"),
                    pl.col("s_sy_customer_birth_country").alias(
                        "customer_birth_country"
                    ),
                ]
            )
            # SQL: ORDER BY customer_id, customer_first_name, customer_last_name, customer_birth_country
            .sort(sort_cols, nulls_last=True)
            # SQL: LIMIT 100
            .limit(100)
        ),
        sort_by=[
            ("customer_id", False),
            ("customer_first_name", False),
            ("customer_last_name", False),
            ("customer_birth_country", False),
        ],
        limit=100,
    )
