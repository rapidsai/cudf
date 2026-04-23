# ruff: noqa: COM812, S608
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 9."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.pdsds_parameters import load_parameters
from cudf_polars.experimental.benchmarks.utils import QueryResult, get_data

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


def polars_impl(run_config: RunConfig) -> QueryResult:
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

    bucket_expressions = []
    for i, (min_qty, max_qty, _) in enumerate(buckets, 1):
        condition = pl.col("ss_quantity").is_between(min_qty, max_qty, closed="both")
        bucket_expressions.extend(
            [
                condition.sum().alias(f"count_{i}"),
                pl.when(condition)
                .then(pl.col(aggcthen))
                .otherwise(None)
                .mean()
                .alias(f"avg_then_{i}"),
                pl.when(condition)
                .then(pl.col(aggcelse))
                .otherwise(None)
                .mean()
                .alias(f"avg_else_{i}"),
            ]
        )

    combined_stats = store_sales.select(bucket_expressions)

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
    return QueryResult(
        frame=(
            reason.filter(pl.col("r_reason_sk") == 1)
            .join(combined_stats, how="cross")
            .select(bucket_values)
            .limit(1)
        ),
        sort_by=[],
        limit=1,
    )


def polars_impl_naive(run_config: RunConfig) -> QueryResult:
    """Query 9 (naive)."""
    params = load_parameters(
        int(run_config.scale_factor), query_id=9, qualification=run_config.qualification
    )

    aggcthen = params["aggcthen"]
    aggcelse = params["aggcelse"]
    rc = params["rc"]

    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    reason = get_data(run_config.dataset_path, "reason", run_config.suffix)

    bucket1_stats = (
        store_sales.filter(pl.col("ss_quantity").is_between(1, 20, closed="both"))
        .select(
            [
                pl.len().alias("count_1"),
                pl.col(aggcthen).mean().alias("avg_then_1"),
                pl.col(aggcelse).mean().alias("avg_else_1"),
            ]
        )
        .with_columns(pl.lit(1).alias("join_key"))
    )
    bucket2_stats = (
        store_sales.filter(pl.col("ss_quantity").is_between(21, 40, closed="both"))
        .select(
            [
                pl.len().alias("count_2"),
                pl.col(aggcthen).mean().alias("avg_then_2"),
                pl.col(aggcelse).mean().alias("avg_else_2"),
            ]
        )
        .with_columns(pl.lit(1).alias("join_key"))
    )
    bucket3_stats = (
        store_sales.filter(pl.col("ss_quantity").is_between(41, 60, closed="both"))
        .select(
            [
                pl.len().alias("count_3"),
                pl.col(aggcthen).mean().alias("avg_then_3"),
                pl.col(aggcelse).mean().alias("avg_else_3"),
            ]
        )
        .with_columns(pl.lit(1).alias("join_key"))
    )
    bucket4_stats = (
        store_sales.filter(pl.col("ss_quantity").is_between(61, 80, closed="both"))
        .select(
            [
                pl.len().alias("count_4"),
                pl.col(aggcthen).mean().alias("avg_then_4"),
                pl.col(aggcelse).mean().alias("avg_else_4"),
            ]
        )
        .with_columns(pl.lit(1).alias("join_key"))
    )
    bucket5_stats = (
        store_sales.filter(pl.col("ss_quantity").is_between(81, 100, closed="both"))
        .select(
            [
                pl.len().alias("count_5"),
                pl.col(aggcthen).mean().alias("avg_then_5"),
                pl.col(aggcelse).mean().alias("avg_else_5"),
            ]
        )
        .with_columns(pl.lit(1).alias("join_key"))
    )

    return QueryResult(
        frame=(
            reason.filter(pl.col("r_reason_sk") == 1)
            .with_columns(pl.lit(1).alias("join_key"))
            .join(bucket1_stats, on="join_key")
            .join(bucket2_stats, on="join_key")
            .join(bucket3_stats, on="join_key")
            .join(bucket4_stats, on="join_key")
            .join(bucket5_stats, on="join_key")
            .select(
                [
                    pl.when(pl.col("count_1") > rc[0])
                    .then(pl.col("avg_then_1"))
                    .otherwise(pl.col("avg_else_1"))
                    .alias("bucket1"),
                    pl.when(pl.col("count_2") > rc[1])
                    .then(pl.col("avg_then_2"))
                    .otherwise(pl.col("avg_else_2"))
                    .alias("bucket2"),
                    pl.when(pl.col("count_3") > rc[2])
                    .then(pl.col("avg_then_3"))
                    .otherwise(pl.col("avg_else_3"))
                    .alias("bucket3"),
                    pl.when(pl.col("count_4") > rc[3])
                    .then(pl.col("avg_then_4"))
                    .otherwise(pl.col("avg_else_4"))
                    .alias("bucket4"),
                    pl.when(pl.col("count_5") > rc[4])
                    .then(pl.col("avg_then_5"))
                    .otherwise(pl.col("avg_else_5"))
                    .alias("bucket5"),
                ]
            )
            .limit(1)
        ),
        sort_by=[],
        limit=1,
    )
