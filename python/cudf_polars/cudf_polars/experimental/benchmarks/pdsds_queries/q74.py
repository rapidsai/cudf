# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 74."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.utils import get_data

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 74."""
    return """
    WITH year_total
         AS (SELECT c_customer_id    customer_id,
                    c_first_name     customer_first_name,
                    c_last_name      customer_last_name,
                    d_year           AS year1,
                    Sum(ss_net_paid) year_total,
                    's'              sale_type
             FROM   customer,
                    store_sales,
                    date_dim
             WHERE  c_customer_sk = ss_customer_sk
                    AND ss_sold_date_sk = d_date_sk
                    AND d_year IN ( 1999, 1999 + 1 )
             GROUP  BY c_customer_id,
                       c_first_name,
                       c_last_name,
                       d_year
             UNION ALL
             SELECT c_customer_id    customer_id,
                    c_first_name     customer_first_name,
                    c_last_name      customer_last_name,
                    d_year           AS year1,
                    Sum(ws_net_paid) year_total,
                    'w'              sale_type
             FROM   customer,
                    web_sales,
                    date_dim
             WHERE  c_customer_sk = ws_bill_customer_sk
                    AND ws_sold_date_sk = d_date_sk
                    AND d_year IN ( 1999, 1999 + 1 )
             GROUP  BY c_customer_id,
                       c_first_name,
                       c_last_name,
                       d_year)
    SELECT t_s_secyear.customer_id,
                   t_s_secyear.customer_first_name,
                   t_s_secyear.customer_last_name
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
           AND t_s_firstyear.year1 = 1999
           AND t_s_secyear.year1 = 1999 + 1
           AND t_w_firstyear.year1 = 1999
           AND t_w_secyear.year1 = 1999 + 1
           AND t_s_firstyear.year_total > 0
           AND t_w_firstyear.year_total > 0
           AND CASE
                 WHEN t_w_firstyear.year_total > 0 THEN t_w_secyear.year_total /
                                                        t_w_firstyear.year_total
                 ELSE NULL
               END > CASE
                       WHEN t_s_firstyear.year_total > 0 THEN
                       t_s_secyear.year_total /
                       t_s_firstyear.year_total
                       ELSE NULL
                     END
    ORDER  BY 1,
              2,
              3
    LIMIT 100;
    """


def _year_totals_component(
    sales: pl.LazyFrame,
    date_dim: pl.LazyFrame,
    customer: pl.LazyFrame,
    date_fk: str,
    customer_fk: str,
    amount_col: str,
    sale_type: str,
) -> pl.LazyFrame:
    dates = date_dim.filter(pl.col("d_year").is_in([1999, 2000])).select(
        ["d_date_sk", "d_year"]
    )
    cust = customer.select(
        ["c_customer_sk", "c_customer_id", "c_first_name", "c_last_name"]
    )
    return (
        sales.join(dates, left_on=date_fk, right_on="d_date_sk")
        .join(cust, left_on=customer_fk, right_on="c_customer_sk")
        .group_by(["c_customer_id", "c_first_name", "c_last_name", "d_year"])
        .agg(pl.col(amount_col).sum().alias("year_total"))
        .select(
            pl.col("c_customer_id").alias("customer_id"),
            pl.col("c_first_name").alias("customer_first_name"),
            pl.col("c_last_name").alias("customer_last_name"),
            pl.col("d_year").alias("year1"),
            "year_total",
        )
        .with_columns(pl.lit(sale_type).alias("sale_type"))
    )


def polars_impl(run_config: RunConfig) -> pl.LazyFrame:
    """Query 74."""
    customer = get_data(run_config.dataset_path, "customer", run_config.suffix)
    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    web_sales = get_data(run_config.dataset_path, "web_sales", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)

    year_total = pl.concat(
        [
            _year_totals_component(
                sales=store_sales,
                date_dim=date_dim,
                customer=customer,
                date_fk="ss_sold_date_sk",
                customer_fk="ss_customer_sk",
                amount_col="ss_net_paid",
                sale_type="s",
            ),
            _year_totals_component(
                sales=web_sales,
                date_dim=date_dim,
                customer=customer,
                date_fk="ws_sold_date_sk",
                customer_fk="ws_bill_customer_sk",
                amount_col="ws_net_paid",
                sale_type="w",
            ),
        ]
    )

    grouped = year_total.group_by(
        ["customer_id", "customer_first_name", "customer_last_name"]
    ).agg(
        [
            (
                pl.when((pl.col("sale_type") == "s") & (pl.col("year1") == 1999))
                .then(pl.col("year_total"))
                .otherwise(None)
            )
            .sum()
            .alias("s_1999"),
            (
                pl.when((pl.col("sale_type") == "s") & (pl.col("year1") == 2000))
                .then(pl.col("year_total"))
                .otherwise(None)
            )
            .sum()
            .alias("s_2000"),
            (
                pl.when((pl.col("sale_type") == "w") & (pl.col("year1") == 1999))
                .then(pl.col("year_total"))
                .otherwise(None)
            )
            .sum()
            .alias("w_1999"),
            (
                pl.when((pl.col("sale_type") == "w") & (pl.col("year1") == 2000))
                .then(pl.col("year_total"))
                .otherwise(None)
            )
            .sum()
            .alias("w_2000"),
        ]
    )

    return (
        grouped.filter((pl.col("s_1999") > 0) & (pl.col("w_1999") > 0))
        .with_columns(
            (pl.col("w_2000") / pl.col("w_1999")).alias("w_ratio"),
            (pl.col("s_2000") / pl.col("s_1999")).alias("s_ratio"),
        )
        .filter(pl.col("w_ratio") > pl.col("s_ratio"))
        .select(["customer_id", "customer_first_name", "customer_last_name"])
        .sort(
            ["customer_id", "customer_first_name", "customer_last_name"],
            nulls_last=True,
        )
        .limit(100)
    )
