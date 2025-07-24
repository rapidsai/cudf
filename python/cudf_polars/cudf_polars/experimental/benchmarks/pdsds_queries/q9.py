# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 9."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.utils import get_data

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 9."""
    return """
    -- start query 9 in stream 0 using template query9.tpl
    SELECT CASE
             WHEN (SELECT Count(*)
                   FROM   store_sales
                   WHERE  ss_quantity BETWEEN 1 AND 20) > 3672 THEN
             (SELECT Avg(ss_ext_list_price)
              FROM   store_sales
              WHERE
             ss_quantity BETWEEN 1 AND 20)
             ELSE (SELECT Avg(ss_net_profit)
                   FROM   store_sales
                   WHERE  ss_quantity BETWEEN 1 AND 20)
           END bucket1,
           CASE
             WHEN (SELECT Count(*)
                   FROM   store_sales
                   WHERE  ss_quantity BETWEEN 21 AND 40) > 3392 THEN
             (SELECT Avg(ss_ext_list_price)
              FROM   store_sales
              WHERE
             ss_quantity BETWEEN 21 AND 40)
             ELSE (SELECT Avg(ss_net_profit)
                   FROM   store_sales
                   WHERE  ss_quantity BETWEEN 21 AND 40)
           END bucket2,
           CASE
             WHEN (SELECT Count(*)
                   FROM   store_sales
                   WHERE  ss_quantity BETWEEN 41 AND 60) > 32784 THEN
             (SELECT Avg(ss_ext_list_price)
              FROM   store_sales
              WHERE
             ss_quantity BETWEEN 41 AND 60)
             ELSE (SELECT Avg(ss_net_profit)
                   FROM   store_sales
                   WHERE  ss_quantity BETWEEN 41 AND 60)
           END bucket3,
           CASE
             WHEN (SELECT Count(*)
                   FROM   store_sales
                   WHERE  ss_quantity BETWEEN 61 AND 80) > 26032 THEN
             (SELECT Avg(ss_ext_list_price)
              FROM   store_sales
              WHERE
             ss_quantity BETWEEN 61 AND 80)
             ELSE (SELECT Avg(ss_net_profit)
                   FROM   store_sales
                   WHERE  ss_quantity BETWEEN 61 AND 80)
           END bucket4,
           CASE
             WHEN (SELECT Count(*)
                   FROM   store_sales
                   WHERE  ss_quantity BETWEEN 81 AND 100) > 23982 THEN
             (SELECT Avg(ss_ext_list_price)
              FROM   store_sales
              WHERE
             ss_quantity BETWEEN 81 AND 100)
             ELSE (SELECT Avg(ss_net_profit)
                   FROM   store_sales
                   WHERE  ss_quantity BETWEEN 81 AND 100)
           END bucket5
    FROM   reason
    WHERE  r_reason_sk = 1;
    """


def polars_impl(run_config: RunConfig) -> pl.LazyFrame:
    """Query 9."""
    # Load required tables
    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    reason = get_data(run_config.dataset_path, "reason", run_config.suffix)

    # Define bucket configurations: (min_qty, max_qty, count_threshold)
    buckets = [
        (1, 20, 3672),
        (21, 40, 3392),
        (41, 60, 32784),
        (61, 80, 26032),
        (81, 100, 23982),
    ]

    # Calculate each bucket value
    bucket_values = []

    for _, (min_qty, max_qty, threshold) in enumerate(buckets, 1):
        # Count records in this quantity range
        count = (
            store_sales.filter(
                pl.col("ss_quantity").is_between(min_qty, max_qty, closed="both")
            )
            .select(pl.len())
            .collect()
            .item()
        )

        # Choose aggregation based on count vs threshold
        if count > threshold:
            # Use average list price
            value = (
                store_sales.filter(
                    pl.col("ss_quantity").is_between(min_qty, max_qty, closed="both")
                )
                .select(pl.col("ss_ext_list_price").mean())
                .collect()
                .item()
            )
        else:
            # Use average net profit
            value = (
                store_sales.filter(
                    pl.col("ss_quantity").is_between(min_qty, max_qty, closed="both")
                )
                .select(pl.col("ss_net_profit").mean())
                .collect()
                .item()
            )

        bucket_values.append(value)

    # Create result DataFrame with one row (using reason table as in SQL)
    return (
        reason.filter(pl.col("r_reason_sk") == 1)
        .select(
            [
                pl.lit(bucket_values[0]).alias("bucket1"),
                pl.lit(bucket_values[1]).alias("bucket2"),
                pl.lit(bucket_values[2]).alias("bucket3"),
                pl.lit(bucket_values[3]).alias("bucket4"),
                pl.lit(bucket_values[4]).alias("bucket5"),
            ]
        )
        .limit(1)
    )
