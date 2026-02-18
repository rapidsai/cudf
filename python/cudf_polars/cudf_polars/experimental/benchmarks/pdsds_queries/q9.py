# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 9."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.pdsds_parameters import load_parameters
from cudf_polars.experimental.benchmarks.utils import get_data

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 9."""
    params = load_parameters(
        int(run_config.scale_factor), query_id=9, qualification=run_config.qualification
    )

    aggcthen = params["aggcthen"]
    aggcelse = params["aggcelse"]
    rc = params["rc"]

    return f"""
    SELECT CASE
             WHEN (SELECT Count(*)
                   FROM   store_sales
                   WHERE  ss_quantity BETWEEN 1 AND 20) > {rc[0]} THEN
             (SELECT Avg({aggcthen})
              FROM   store_sales
              WHERE
             ss_quantity BETWEEN 1 AND 20)
             ELSE (SELECT Avg({aggcelse})
                   FROM   store_sales
                   WHERE  ss_quantity BETWEEN 1 AND 20)
           END bucket1,
           CASE
             WHEN (SELECT Count(*)
                   FROM   store_sales
                   WHERE  ss_quantity BETWEEN 21 AND 40) > {rc[1]} THEN
             (SELECT Avg({aggcthen})
              FROM   store_sales
              WHERE
             ss_quantity BETWEEN 21 AND 40)
             ELSE (SELECT Avg({aggcelse})
                   FROM   store_sales
                   WHERE  ss_quantity BETWEEN 21 AND 40)
           END bucket2,
           CASE
             WHEN (SELECT Count(*)
                   FROM   store_sales
                   WHERE  ss_quantity BETWEEN 41 AND 60) > {rc[2]} THEN
             (SELECT Avg({aggcthen})
              FROM   store_sales
              WHERE
             ss_quantity BETWEEN 41 AND 60)
             ELSE (SELECT Avg({aggcelse})
                   FROM   store_sales
                   WHERE  ss_quantity BETWEEN 41 AND 60)
           END bucket3,
           CASE
             WHEN (SELECT Count(*)
                   FROM   store_sales
                   WHERE  ss_quantity BETWEEN 61 AND 80) > {rc[3]} THEN
             (SELECT Avg({aggcthen})
              FROM   store_sales
              WHERE
             ss_quantity BETWEEN 61 AND 80)
             ELSE (SELECT Avg({aggcelse})
                   FROM   store_sales
                   WHERE  ss_quantity BETWEEN 61 AND 80)
           END bucket4,
           CASE
             WHEN (SELECT Count(*)
                   FROM   store_sales
                   WHERE  ss_quantity BETWEEN 81 AND 100) > {rc[4]} THEN
             (SELECT Avg({aggcthen})
              FROM   store_sales
              WHERE
             ss_quantity BETWEEN 81 AND 100)
             ELSE (SELECT Avg({aggcelse})
                   FROM   store_sales
                   WHERE  ss_quantity BETWEEN 81 AND 100)
           END bucket5
    FROM   reason
    WHERE  r_reason_sk = 1;
    """


def polars_impl(run_config: RunConfig) -> pl.LazyFrame:
    """Query 9."""
    params = load_parameters(
        int(run_config.scale_factor), query_id=9, qualification=run_config.qualification
    )

    aggcthen = params["aggcthen"]
    aggcelse = params["aggcelse"]
    rc = params["rc"]

    # Load required tables
    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    reason = get_data(run_config.dataset_path, "reason", run_config.suffix)

    # Define bucket configurations: (min_qty, max_qty, count_threshold)
    buckets = [
        (1, 20, rc[0]),
        (21, 40, rc[1]),
        (41, 60, rc[2]),
        (61, 80, rc[3]),
        (81, 100, rc[4]),
    ]

    # Calculate each bucket summary
    bucket_stats = []
    for i, (min_qty, max_qty, _) in enumerate(buckets, 1):
        # Compute count, avg(aggcthen), avg(aggcelse) for each quantity range
        stats = store_sales.filter(
            pl.col("ss_quantity").is_between(min_qty, max_qty, closed="both")
        ).select(
            [
                pl.len().alias(f"count_{i}"),
                pl.col(aggcthen).mean().alias(f"avg_then_{i}"),
                pl.col(aggcelse).mean().alias(f"avg_else_{i}"),
            ]
        )
        bucket_stats.append(stats)

    # Combine all bucket summaries into one row
    combined_stats = pl.concat(bucket_stats, how="horizontal")

    # Select appropriate value per bucket based on count threshold
    bucket_values = []
    for i, (_min_qty, _max_qty, threshold) in enumerate(buckets, 1):
        bucket = (
            pl.when(pl.col(f"count_{i}") > threshold)
            .then(pl.col(f"avg_then_{i}"))
            .otherwise(pl.col(f"avg_else_{i}"))
            .alias(f"bucket{i}")
        )
        bucket_values.append(bucket)

    # Create result DataFrame with one row (using reason table as in SQL)
    return (
        reason.filter(pl.col("r_reason_sk") == 1)
        .join(combined_stats, how="cross")
        .select(bucket_values)
        .limit(1)
    )
