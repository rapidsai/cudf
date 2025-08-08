# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 4."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.utils import get_data

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 4."""
    return """
    WITH year_total
         AS (SELECT c_customer_id                       customer_id,
                    c_first_name                        customer_first_name,
                    c_last_name                         customer_last_name,
                    c_preferred_cust_flag               customer_preferred_cust_flag
                    ,
                    c_birth_country
                    customer_birth_country,
                    c_login                             customer_login,
                    c_email_address                     customer_email_address,
                    d_year                              dyear,
                    Sum(( ( ss_ext_list_price - ss_ext_wholesale_cost
                            - ss_ext_discount_amt
                          )
                          +
                              ss_ext_sales_price ) / 2) year_total,
                    's'                                 sale_type
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
             SELECT c_customer_id                             customer_id,
                    c_first_name                              customer_first_name,
                    c_last_name                               customer_last_name,
                    c_preferred_cust_flag
                    customer_preferred_cust_flag,
                    c_birth_country                           customer_birth_country
                    ,
                    c_login
                    customer_login,
                    c_email_address                           customer_email_address
                    ,
                    d_year                                    dyear
                    ,
                    Sum(( ( ( cs_ext_list_price
                              - cs_ext_wholesale_cost
                              - cs_ext_discount_amt
                            ) +
                                  cs_ext_sales_price ) / 2 )) year_total,
                    'c'                                       sale_type
             FROM   customer,
                    catalog_sales,
                    date_dim
             WHERE  c_customer_sk = cs_bill_customer_sk
                    AND cs_sold_date_sk = d_date_sk
             GROUP  BY c_customer_id,
                       c_first_name,
                       c_last_name,
                       c_preferred_cust_flag,
                       c_birth_country,
                       c_login,
                       c_email_address,
                       d_year
             UNION ALL
             SELECT c_customer_id                             customer_id,
                    c_first_name                              customer_first_name,
                    c_last_name                               customer_last_name,
                    c_preferred_cust_flag
                    customer_preferred_cust_flag,
                    c_birth_country                           customer_birth_country
                    ,
                    c_login
                    customer_login,
                    c_email_address                           customer_email_address
                    ,
                    d_year                                    dyear
                    ,
                    Sum(( ( ( ws_ext_list_price
                              - ws_ext_wholesale_cost
                              - ws_ext_discount_amt
                            ) +
                                  ws_ext_sales_price ) / 2 )) year_total,
                    'w'                                       sale_type
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
                       d_year)
    SELECT t_s_secyear.customer_id,
                   t_s_secyear.customer_first_name,
                   t_s_secyear.customer_last_name,
                   t_s_secyear.customer_preferred_cust_flag
    FROM   year_total t_s_firstyear,
           year_total t_s_secyear,
           year_total t_c_firstyear,
           year_total t_c_secyear,
           year_total t_w_firstyear,
           year_total t_w_secyear
    WHERE  t_s_secyear.customer_id = t_s_firstyear.customer_id
           AND t_s_firstyear.customer_id = t_c_secyear.customer_id
           AND t_s_firstyear.customer_id = t_c_firstyear.customer_id
           AND t_s_firstyear.customer_id = t_w_firstyear.customer_id
           AND t_s_firstyear.customer_id = t_w_secyear.customer_id
           AND t_s_firstyear.sale_type = 's'
           AND t_c_firstyear.sale_type = 'c'
           AND t_w_firstyear.sale_type = 'w'
           AND t_s_secyear.sale_type = 's'
           AND t_c_secyear.sale_type = 'c'
           AND t_w_secyear.sale_type = 'w'
           AND t_s_firstyear.dyear = 2001
           AND t_s_secyear.dyear = 2001 + 1
           AND t_c_firstyear.dyear = 2001
           AND t_c_secyear.dyear = 2001 + 1
           AND t_w_firstyear.dyear = 2001
           AND t_w_secyear.dyear = 2001 + 1
           AND t_s_firstyear.year_total > 0
           AND t_c_firstyear.year_total > 0
           AND t_w_firstyear.year_total > 0
           AND CASE
                 WHEN t_c_firstyear.year_total > 0 THEN t_c_secyear.year_total /
                                                        t_c_firstyear.year_total
                 ELSE NULL
               END > CASE
                       WHEN t_s_firstyear.year_total > 0 THEN
                       t_s_secyear.year_total /
                       t_s_firstyear.year_total
                       ELSE NULL
                     END
           AND CASE
                 WHEN t_c_firstyear.year_total > 0 THEN t_c_secyear.year_total /
                                                        t_c_firstyear.year_total
                 ELSE NULL
               END > CASE
                       WHEN t_w_firstyear.year_total > 0 THEN
                       t_w_secyear.year_total /
                       t_w_firstyear.year_total
                       ELSE NULL
                     END
    ORDER  BY t_s_secyear.customer_id,
              t_s_secyear.customer_first_name,
              t_s_secyear.customer_last_name,
              t_s_secyear.customer_preferred_cust_flag
    LIMIT 100;
    """


def build_sales_subquery(  # noqa: D103
    sales_df: pl.LazyFrame,
    date_df: pl.LazyFrame,
    customer_df: pl.LazyFrame,
    sold_date_key: str,
    customer_key: str,
    col_prefix: str,
    *,
    year_filter: bool = False,
    include_customer_info: bool = False,
) -> pl.LazyFrame:
    profit_expr = (
        (
            pl.col(f"{col_prefix}ext_list_price")
            - pl.col(f"{col_prefix}ext_wholesale_cost")
            - pl.col(f"{col_prefix}ext_discount_amt")
        )
        + pl.col(f"{col_prefix}ext_sales_price")
    ) / 2

    df = (
        sales_df.join(date_df, left_on=sold_date_key, right_on="d_date_sk")
        .join(customer_df, left_on=customer_key, right_on="c_customer_sk")
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
        .agg(profit_expr.sum().alias("year_total"))
    )

    if year_filter:
        df = df.filter(pl.col("year_total") > 0)

    if include_customer_info:
        return df.select(
            [
                pl.col("c_customer_id").alias("customer_id"),
                pl.col("c_first_name").alias("customer_first_name"),
                pl.col("c_last_name").alias("customer_last_name"),
                pl.col("c_preferred_cust_flag").alias("customer_preferred_cust_flag"),
                pl.col("year_total"),
            ]
        )
    else:
        return df.select(
            [pl.col("c_customer_id").alias("customer_id"), pl.col("year_total")]
        )


def polars_impl(run_config: RunConfig) -> pl.LazyFrame:
    """Query 4."""
    # Load required tables
    customer = get_data(run_config.dataset_path, "customer", run_config.suffix)
    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    catalog_sales = get_data(
        run_config.dataset_path, "catalog_sales", run_config.suffix
    )
    web_sales = get_data(run_config.dataset_path, "web_sales", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    date_2001 = date_dim.filter(pl.col("d_year") == 2001)
    date_2002 = date_dim.filter(pl.col("d_year") == 2002)

    # Store sales - first year (2001)
    t_s_firstyear = build_sales_subquery(
        store_sales,
        date_2001,
        customer,
        sold_date_key="ss_sold_date_sk",
        customer_key="ss_customer_sk",
        col_prefix="ss_",
        year_filter=True,
        include_customer_info=True,
    )

    # Store sales - second year (2002)
    t_s_secyear = build_sales_subquery(
        store_sales,
        date_2002,
        customer,
        sold_date_key="ss_sold_date_sk",
        customer_key="ss_customer_sk",
        col_prefix="ss_",
        year_filter=False,
        include_customer_info=True,
    )

    # Catalog sales - first year (2001)
    t_c_firstyear = build_sales_subquery(
        catalog_sales,
        date_2001,
        customer,
        sold_date_key="cs_sold_date_sk",
        customer_key="cs_bill_customer_sk",
        col_prefix="cs_",
        year_filter=True,
        include_customer_info=False,
    )

    # Catalog sales - first year (2002)
    t_c_secyear = build_sales_subquery(
        catalog_sales,
        date_2002,
        customer,
        sold_date_key="cs_sold_date_sk",
        customer_key="cs_bill_customer_sk",
        col_prefix="cs_",
        year_filter=False,
        include_customer_info=False,
    )

    # Web sales - first year (2001)
    t_w_firstyear = build_sales_subquery(
        web_sales,
        date_2001,
        customer,
        sold_date_key="ws_sold_date_sk",
        customer_key="ws_bill_customer_sk",
        col_prefix="ws_",
        year_filter=True,
        include_customer_info=False,
    )

    # Web sales - first year (2001)
    t_w_secyear = build_sales_subquery(
        web_sales,
        date_2002,
        customer,
        sold_date_key="ws_sold_date_sk",
        customer_key="ws_bill_customer_sk",
        col_prefix="ws_",
        year_filter=False,
        include_customer_info=False,
    )

    # Perform the joins and filtering
    sort_cols = [
        "customer_id",
        "customer_first_name",
        "customer_last_name",
        "customer_preferred_cust_flag",
    ]
    return (
        t_s_secyear.join(t_s_firstyear, on="customer_id", suffix="_sf", how="inner")
        .join(t_c_firstyear, on="customer_id", suffix="_cf", how="inner")
        .join(t_c_secyear, on="customer_id", suffix="_cs", how="inner")
        .join(t_w_firstyear, on="customer_id", suffix="_wf", how="inner")
        .join(t_w_secyear, on="customer_id", suffix="_ws", how="inner")
        .filter(
            # All first year totals must be > 0
            (pl.col("year_total_sf") > 0)
            & (pl.col("year_total_cf") > 0)
            & (pl.col("year_total_wf") > 0)
            &
            # Catalog growth rate > Store growth rate
            (
                pl.when(pl.col("year_total_cf") > 0)
                .then(pl.col("year_total_cs") / pl.col("year_total_cf"))
                .otherwise(None)
                > pl.when(pl.col("year_total_sf") > 0)
                .then(pl.col("year_total") / pl.col("year_total_sf"))
                .otherwise(None)
            )
            &
            # Catalog growth rate > Web growth rate
            (
                pl.when(pl.col("year_total_cf") > 0)
                .then(pl.col("year_total_cs") / pl.col("year_total_cf"))
                .otherwise(None)
                > pl.when(pl.col("year_total_wf") > 0)
                .then(pl.col("year_total_ws") / pl.col("year_total_wf"))
                .otherwise(None)
            )
        )
        .select(sort_cols)
        .sort(sort_cols)
        .limit(100)
    )
