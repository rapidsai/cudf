# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 28."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.utils import get_data

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 28."""
    return """
    SELECT *
    FROM   (SELECT Avg(ss_list_price)            B1_LP,
                   Count(ss_list_price)          B1_CNT,
                   Count(DISTINCT ss_list_price) B1_CNTD
            FROM   store_sales
            WHERE  ss_quantity BETWEEN 0 AND 5
                   AND ( ss_list_price BETWEEN 18 AND 18 + 10
                          OR ss_coupon_amt BETWEEN 1939 AND 1939 + 1000
                          OR ss_wholesale_cost BETWEEN 34 AND 34 + 20 )) B1,
           (SELECT Avg(ss_list_price)            B2_LP,
                   Count(ss_list_price)          B2_CNT,
                   Count(DISTINCT ss_list_price) B2_CNTD
            FROM   store_sales
            WHERE  ss_quantity BETWEEN 6 AND 10
                   AND ( ss_list_price BETWEEN 1 AND 1 + 10
                          OR ss_coupon_amt BETWEEN 35 AND 35 + 1000
                          OR ss_wholesale_cost BETWEEN 50 AND 50 + 20 )) B2,
           (SELECT Avg(ss_list_price)            B3_LP,
                   Count(ss_list_price)          B3_CNT,
                   Count(DISTINCT ss_list_price) B3_CNTD
            FROM   store_sales
            WHERE  ss_quantity BETWEEN 11 AND 15
                   AND ( ss_list_price BETWEEN 91 AND 91 + 10
                          OR ss_coupon_amt BETWEEN 1412 AND 1412 + 1000
                          OR ss_wholesale_cost BETWEEN 17 AND 17 + 20 )) B3,
           (SELECT Avg(ss_list_price)            B4_LP,
                   Count(ss_list_price)          B4_CNT,
                   Count(DISTINCT ss_list_price) B4_CNTD
            FROM   store_sales
            WHERE  ss_quantity BETWEEN 16 AND 20
                   AND ( ss_list_price BETWEEN 9 AND 9 + 10
                          OR ss_coupon_amt BETWEEN 5270 AND 5270 + 1000
                          OR ss_wholesale_cost BETWEEN 29 AND 29 + 20 )) B4,
           (SELECT Avg(ss_list_price)            B5_LP,
                   Count(ss_list_price)          B5_CNT,
                   Count(DISTINCT ss_list_price) B5_CNTD
            FROM   store_sales
            WHERE  ss_quantity BETWEEN 21 AND 25
                   AND ( ss_list_price BETWEEN 45 AND 45 + 10
                          OR ss_coupon_amt BETWEEN 826 AND 826 + 1000
                          OR ss_wholesale_cost BETWEEN 5 AND 5 + 20 )) B5,
           (SELECT Avg(ss_list_price)            B6_LP,
                   Count(ss_list_price)          B6_CNT,
                   Count(DISTINCT ss_list_price) B6_CNTD
            FROM   store_sales
            WHERE  ss_quantity BETWEEN 26 AND 30
                   AND ( ss_list_price BETWEEN 174 AND 174 + 10
                          OR ss_coupon_amt BETWEEN 5548 AND 5548 + 1000
                          OR ss_wholesale_cost BETWEEN 42 AND 42 + 20 )) B6
    LIMIT 100;
    """


def make_block(
    store_sales: pl.LazyFrame,
    lp_min: int,
    lp_max: int,
    ca_min: int,
    ca_max: int,
    wc_min: int,
    wc_max: int,
    q_lo: int,
    q_hi: int,
    prefix: str,
) -> pl.LazyFrame:
    """Make store sales filter block."""
    return store_sales.filter(
        pl.col("ss_quantity").is_between(q_lo, q_hi)
        & (
            pl.col("ss_list_price").is_between(lp_min, lp_max)
            | pl.col("ss_coupon_amt").is_between(ca_min, ca_max)
            | pl.col("ss_wholesale_cost").is_between(wc_min, wc_max)
        )
    ).select(
        [
            pl.col("ss_list_price").mean().alias(f"{prefix}_LP"),
            pl.col("ss_list_price")
            .drop_nulls()
            .count()
            .cast(pl.Int64)
            .alias(f"{prefix}_CNT"),
            pl.col("ss_list_price")
            .drop_nulls()
            .n_unique()
            .cast(pl.Int64)
            .alias(f"{prefix}_CNTD"),
        ]
    )


def polars_impl(run_config: RunConfig) -> pl.LazyFrame:
    """Query 28."""
    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)

    b1 = make_block(store_sales, 18, 28, 1939, 2939, 34, 54, 0, 5, "B1")
    b2 = make_block(store_sales, 1, 11, 35, 1035, 50, 70, 6, 10, "B2")
    b3 = make_block(store_sales, 91, 101, 1412, 2412, 17, 37, 11, 15, "B3")
    b4 = make_block(store_sales, 9, 19, 5270, 6270, 29, 49, 16, 20, "B4")
    b5 = make_block(store_sales, 45, 55, 826, 1826, 5, 25, 21, 25, "B5")
    b6 = make_block(store_sales, 174, 184, 5548, 6548, 42, 62, 26, 30, "B6")

    return (
        b1.join(b2, how="cross")
        .join(b3, how="cross")
        .join(b4, how="cross")
        .join(b5, how="cross")
        .join(b6, how="cross")
        .limit(100)
    )
