# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 50."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.utils import get_data

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 50."""
    return """
    SELECT s_store_name,
                   s_company_id,
                   s_street_number,
                   s_street_name,
                   s_street_type,
                   s_suite_number,
                   s_city,
                   s_county,
                   s_state,
                   s_zip,
                   Sum(CASE
                         WHEN ( sr_returned_date_sk - ss_sold_date_sk <= 30 ) THEN 1
                         ELSE 0
                       END) AS '30 days',
                   Sum(CASE
                         WHEN ( sr_returned_date_sk - ss_sold_date_sk > 30 )
                              AND ( sr_returned_date_sk - ss_sold_date_sk <= 60 )
                       THEN 1
                         ELSE 0
                       END) AS '31-60 days',
                   Sum(CASE
                         WHEN ( sr_returned_date_sk - ss_sold_date_sk > 60 )
                              AND ( sr_returned_date_sk - ss_sold_date_sk <= 90 )
                       THEN 1
                         ELSE 0
                       END) AS '61-90 days',
                   Sum(CASE
                         WHEN ( sr_returned_date_sk - ss_sold_date_sk > 90 )
                              AND ( sr_returned_date_sk - ss_sold_date_sk <= 120 )
                       THEN 1
                         ELSE 0
                       END) AS '91-120 days',
                   Sum(CASE
                         WHEN ( sr_returned_date_sk - ss_sold_date_sk > 120 ) THEN 1
                         ELSE 0
                       END) AS '>120 days'
    FROM   store_sales,
           store_returns,
           store,
           date_dim d1,
           date_dim d2
    WHERE  d2.d_year = 2002
           AND d2.d_moy = 9
           AND ss_ticket_number = sr_ticket_number
           AND ss_item_sk = sr_item_sk
           AND ss_sold_date_sk = d1.d_date_sk
           AND sr_returned_date_sk = d2.d_date_sk
           AND ss_customer_sk = sr_customer_sk
           AND ss_store_sk = s_store_sk
    GROUP  BY s_store_name,
              s_company_id,
              s_street_number,
              s_street_name,
              s_street_type,
              s_suite_number,
              s_city,
              s_county,
              s_state,
              s_zip
    ORDER  BY s_store_name,
              s_company_id,
              s_street_number,
              s_street_name,
              s_street_type,
              s_suite_number,
              s_city,
              s_county,
              s_state,
              s_zip
    LIMIT 100;
    """


def polars_impl(run_config: RunConfig) -> pl.LazyFrame:
    """Query 50."""
    # Load tables
    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    store_returns = get_data(
        run_config.dataset_path, "store_returns", run_config.suffix
    )
    store = get_data(run_config.dataset_path, "store", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    return (
        store_sales
        # Join with store_returns on ticket_number, item_sk, and customer_sk
        .join(
            store_returns,
            left_on=["ss_ticket_number", "ss_item_sk", "ss_customer_sk"],
            right_on=["sr_ticket_number", "sr_item_sk", "sr_customer_sk"],
            how="inner",
        )
        # Join with store for store details
        .join(store, left_on="ss_store_sk", right_on="s_store_sk")
        # Join with date_dim for sales date (d1)
        .join(
            date_dim.select(
                [
                    pl.col("d_date_sk").alias("d1_date_sk"),
                    pl.col("d_year").alias("d1_year"),
                    pl.col("d_moy").alias("d1_moy"),
                ]
            ),
            left_on="ss_sold_date_sk",
            right_on="d1_date_sk",
        )
        # Join with date_dim for return date (d2)
        .join(
            date_dim.select(
                [
                    pl.col("d_date_sk").alias("d2_date_sk"),
                    pl.col("d_year").alias("d2_year"),
                    pl.col("d_moy").alias("d2_moy"),
                ]
            ),
            left_on="sr_returned_date_sk",
            right_on="d2_date_sk",
        )
        # Filter for returns in 2002-09
        .filter((pl.col("d2_year") == 2002) & (pl.col("d2_moy") == 9))
        # Pre-compute bucket indicator columns
        .with_columns(
            [
                ((pl.col("sr_returned_date_sk") - pl.col("ss_sold_date_sk")) <= 30)
                .cast(pl.Int32)
                .alias("bucket_30"),
                (
                    ((pl.col("sr_returned_date_sk") - pl.col("ss_sold_date_sk")) > 30)
                    & (
                        (pl.col("sr_returned_date_sk") - pl.col("ss_sold_date_sk"))
                        <= 60
                    )
                )
                .cast(pl.Int32)
                .alias("bucket_31_60"),
                (
                    ((pl.col("sr_returned_date_sk") - pl.col("ss_sold_date_sk")) > 60)
                    & (
                        (pl.col("sr_returned_date_sk") - pl.col("ss_sold_date_sk"))
                        <= 90
                    )
                )
                .cast(pl.Int32)
                .alias("bucket_61_90"),
                (
                    ((pl.col("sr_returned_date_sk") - pl.col("ss_sold_date_sk")) > 90)
                    & (
                        (pl.col("sr_returned_date_sk") - pl.col("ss_sold_date_sk"))
                        <= 120
                    )
                )
                .cast(pl.Int32)
                .alias("bucket_91_120"),
                ((pl.col("sr_returned_date_sk") - pl.col("ss_sold_date_sk")) > 120)
                .cast(pl.Int32)
                .alias("bucket_120_plus"),
            ]
        )
        # Calculate return time buckets with simple aggregations
        .group_by(
            [
                "s_store_name",
                "s_company_id",
                "s_street_number",
                "s_street_name",
                "s_street_type",
                "s_suite_number",
                "s_city",
                "s_county",
                "s_state",
                "s_zip",
            ]
        )
        .agg(
            [
                pl.col("bucket_30").sum().alias("bucket_30_sum"),
                pl.col("bucket_31_60").sum().alias("bucket_31_60_sum"),
                pl.col("bucket_61_90").sum().alias("bucket_61_90_sum"),
                pl.col("bucket_91_120").sum().alias("bucket_91_120_sum"),
                pl.col("bucket_120_plus").sum().alias("bucket_120_plus_sum"),
                pl.col("sr_returned_date_sk").count().alias("return_count"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("return_count") > 0)
                .then(pl.col("bucket_30_sum"))
                .otherwise(None)
                .alias("30 days"),
                pl.when(pl.col("return_count") > 0)
                .then(pl.col("bucket_31_60_sum"))
                .otherwise(None)
                .alias("31-60 days"),
                pl.when(pl.col("return_count") > 0)
                .then(pl.col("bucket_61_90_sum"))
                .otherwise(None)
                .alias("61-90 days"),
                pl.when(pl.col("return_count") > 0)
                .then(pl.col("bucket_91_120_sum"))
                .otherwise(None)
                .alias("91-120 days"),
                pl.when(pl.col("return_count") > 0)
                .then(pl.col("bucket_120_plus_sum"))
                .otherwise(None)
                .alias(">120 days"),
            ]
        )
        .drop(
            [
                "bucket_30_sum",
                "bucket_31_60_sum",
                "bucket_61_90_sum",
                "bucket_91_120_sum",
                "bucket_120_plus_sum",
                "return_count",
            ]
        )
        .select(
            [
                "s_store_name",
                "s_company_id",
                "s_street_number",
                "s_street_name",
                "s_street_type",
                "s_suite_number",
                "s_city",
                "s_county",
                "s_state",
                "s_zip",
                "30 days",
                "31-60 days",
                "61-90 days",
                "91-120 days",
                ">120 days",
            ]
        )
        .sort(
            [
                "s_store_name",
                "s_company_id",
                "s_street_number",
                "s_street_name",
                "s_street_type",
                "s_suite_number",
                "s_city",
                "s_county",
                "s_state",
                "s_zip",
            ],
            nulls_last=True,
            descending=[False] * 10,
        )
        .limit(100)
    )
