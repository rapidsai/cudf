# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 40."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.pdsds_parameters import load_parameters
from cudf_polars.experimental.benchmarks.utils import QueryResult, get_data

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 40."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=40,
        qualification=run_config.qualification,
    )

    sales_date = params["sales_date"]

    return f"""
    SELECT
                    w_state ,
                    i_item_id ,
                    Sum(
                    CASE
                                    WHEN (
                                                                    Cast(d_date AS DATE) < Cast ('{sales_date}' AS DATE)) THEN cs_sales_price - COALESCE(cr_refunded_cash,0)
                                    ELSE 0
                    END) AS sales_before ,
                    Sum(
                    CASE
                                    WHEN (
                                                                    Cast(d_date AS DATE) >= Cast ('{sales_date}' AS DATE)) THEN cs_sales_price - COALESCE(cr_refunded_cash,0)
                                    ELSE 0
                    END) AS sales_after
    FROM            catalog_sales
    LEFT OUTER JOIN catalog_returns
    ON              (
                                    cs_order_number = cr_order_number
                    AND             cs_item_sk = cr_item_sk) ,
                    warehouse ,
                    item ,
                    date_dim
    WHERE           i_current_price BETWEEN 0.99 AND             1.49
    AND             i_item_sk = cs_item_sk
    AND             cs_warehouse_sk = w_warehouse_sk
    AND             cs_sold_date_sk = d_date_sk
    AND             d_date BETWEEN (Cast ('{sales_date}' AS DATE) - INTERVAL '30' day) AND             (
                                    cast ('{sales_date}' AS date) + INTERVAL '30' day)
    GROUP BY        w_state,
                    i_item_id
    ORDER BY        w_state,
                    i_item_id
    LIMIT 100;
    """


def polars_impl(run_config: RunConfig) -> QueryResult:
    """Query 40."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=40,
        qualification=run_config.qualification,
    )

    sales_date = params["sales_date"]

    sales_date_obj = datetime.strptime(sales_date, "%Y-%m-%d")
    start_date_obj = sales_date_obj - timedelta(days=30)
    end_date_obj = sales_date_obj + timedelta(days=30)

    target_date = pl.date(sales_date_obj.year, sales_date_obj.month, sales_date_obj.day)
    start_date = pl.date(start_date_obj.year, start_date_obj.month, start_date_obj.day)
    end_date = pl.date(end_date_obj.year, end_date_obj.month, end_date_obj.day)

    # Load tables
    catalog_sales = get_data(
        run_config.dataset_path, "catalog_sales", run_config.suffix
    )
    catalog_returns = get_data(
        run_config.dataset_path, "catalog_returns", run_config.suffix
    )
    warehouse = get_data(run_config.dataset_path, "warehouse", run_config.suffix)
    item = get_data(run_config.dataset_path, "item", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    sort_by = {"w_state": False, "i_item_id": False}
    limit = 100
    # Define the target date and date range
    return QueryResult(
        frame=(
            catalog_sales.join(
                catalog_returns,
                left_on=["cs_order_number", "cs_item_sk"],
                right_on=["cr_order_number", "cr_item_sk"],
                how="left",
            )  # LEFT OUTER JOIN
            .join(warehouse, left_on="cs_warehouse_sk", right_on="w_warehouse_sk")
            .join(item, left_on="cs_item_sk", right_on="i_item_sk")
            .join(date_dim, left_on="cs_sold_date_sk", right_on="d_date_sk")
            .filter(
                (pl.col("i_current_price").is_between(0.99, 1.49))
                & (pl.col("d_date").is_between(start_date, end_date))
            )
            .with_columns(
                [
                    pl.when(pl.col("d_date") < target_date)
                    .then(
                        pl.col("cs_sales_price")
                        - pl.col("cr_refunded_cash").fill_null(0)
                    )
                    .otherwise(0)
                    .alias("sales_before_amount"),
                    pl.when(pl.col("d_date") >= target_date)
                    .then(
                        pl.col("cs_sales_price")
                        - pl.col("cr_refunded_cash").fill_null(0)
                    )
                    .otherwise(0)
                    .alias("sales_after_amount"),
                ]
            )
            .group_by(["w_state", "i_item_id"])
            .agg(
                [
                    pl.col("sales_before_amount").sum().alias("sales_before"),
                    pl.col("sales_after_amount").sum().alias("sales_after"),
                ]
            )
            .sort(sort_by.keys(), nulls_last=True)
            .limit(limit)
        ),
        sort_by=list(sort_by.items()),
        limit=limit,
    )


def polars_impl_naive(run_config: RunConfig) -> QueryResult:
    """Query 40 (naive)."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=40,
        qualification=run_config.qualification,
    )

    sales_date = params["sales_date"]

    catalog_sales = get_data(
        run_config.dataset_path, "catalog_sales", run_config.suffix
    )
    catalog_returns = get_data(
        run_config.dataset_path, "catalog_returns", run_config.suffix
    )
    warehouse = get_data(run_config.dataset_path, "warehouse", run_config.suffix)
    item = get_data(run_config.dataset_path, "item", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)

    target_date = pl.lit(sales_date).str.strptime(pl.Date, "%Y-%m-%d")
    start_date = target_date - pl.duration(days=30)
    end_date = target_date + pl.duration(days=30)

    sort_by = {"w_state": False, "i_item_id": False}
    limit = 100
    return QueryResult(
        frame=(
            # SQL: FROM catalog_sales LEFT OUTER JOIN catalog_returns ON (cs_order_number=cr_order_number AND cs_item_sk=cr_item_sk)
            catalog_sales.join(
                catalog_returns,
                left_on=["cs_order_number", "cs_item_sk"],
                right_on=["cr_order_number", "cr_item_sk"],
                how="left",
            )
            # SQL: JOIN warehouse ON cs_warehouse_sk = w_warehouse_sk
            .join(warehouse, left_on="cs_warehouse_sk", right_on="w_warehouse_sk")
            # SQL: JOIN item ON i_item_sk = cs_item_sk
            .join(item, left_on="cs_item_sk", right_on="i_item_sk")
            # SQL: JOIN date_dim ON cs_sold_date_sk = d_date_sk
            .join(date_dim, left_on="cs_sold_date_sk", right_on="d_date_sk")
            # SQL: WHERE i_current_price BETWEEN 0.99 AND 1.49 AND d_date BETWEEN '{sales_date}'-30d AND '{sales_date}'+30d
            .filter(
                pl.col("i_current_price").is_between(0.99, 1.49)
                & pl.col("d_date").is_between(start_date, end_date)
            )
            # SQL: GROUP BY w_state, i_item_id
            .group_by(["w_state", "i_item_id"])
            # SQL: Sum(CASE WHEN d_date < '{sales_date}' THEN cs_sales_price - COALESCE(cr_refunded_cash,0) ELSE 0 END) AS sales_before, Sum(CASE WHEN d_date >= ... THEN ... END) AS sales_after
            .agg(
                [
                    pl.when(pl.col("d_date") < target_date)
                    .then(
                        pl.col("cs_sales_price")
                        - pl.coalesce([pl.col("cr_refunded_cash"), pl.lit(0)])
                    )
                    .otherwise(0)
                    .sum()
                    .alias("sales_before"),
                    pl.when(pl.col("d_date") >= target_date)
                    .then(
                        pl.col("cs_sales_price")
                        - pl.coalesce([pl.col("cr_refunded_cash"), pl.lit(0)])
                    )
                    .otherwise(0)
                    .sum()
                    .alias("sales_after"),
                ]
            )
            # SQL: ORDER BY w_state, i_item_id
            .sort(sort_by.keys(), nulls_last=True)
            # SQL: LIMIT 100
            .limit(limit)
        ),
        sort_by=list(sort_by.items()),
        limit=limit,
    )
