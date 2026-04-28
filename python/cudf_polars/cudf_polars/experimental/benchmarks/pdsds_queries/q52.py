# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 52."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.pdsds_parameters import load_parameters
from cudf_polars.experimental.benchmarks.utils import QueryResult, get_data

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 52."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=52,
        qualification=run_config.qualification,
    )

    year = params["year"]
    month = params["month"]
    manager_id = params["manager_id"]

    return f"""
        SELECT dt.d_year,
                    item.i_brand_id         brand_id,
                    item.i_brand            brand,
                    Sum(ss_ext_sales_price) ext_price
        FROM   date_dim dt,
            store_sales,
            item
        WHERE  dt.d_date_sk = store_sales.ss_sold_date_sk
            AND store_sales.ss_item_sk = item.i_item_sk
            AND item.i_manager_id = {manager_id}
            AND dt.d_moy = {month}
            AND dt.d_year = {year}
        GROUP  BY dt.d_year,
                item.i_brand,
                item.i_brand_id
        ORDER  BY dt.d_year,
                ext_price DESC,
                brand_id
        LIMIT 100;
    """


def polars_impl(run_config: RunConfig) -> QueryResult:
    """Query 52."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=52,
        qualification=run_config.qualification,
    )

    year = params["year"]
    month = params["month"]
    manager_id = params["manager_id"]

    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    item = get_data(run_config.dataset_path, "item", run_config.suffix)

    sort_by = {"d_year": False, "ext_price": True, "brand_id": False}
    limit = 100
    return QueryResult(
        frame=(
            store_sales.join(date_dim, left_on="ss_sold_date_sk", right_on="d_date_sk")
            .join(item, left_on="ss_item_sk", right_on="i_item_sk")
            .filter(
                (pl.col("i_manager_id") == manager_id)
                & (pl.col("d_moy") == month)
                & (pl.col("d_year") == year)
            )
            .group_by(["d_year", "i_brand", "i_brand_id"])
            .agg(pl.col("ss_ext_sales_price").sum().alias("ext_price"))
            .select(
                [
                    pl.col("d_year"),
                    pl.col("i_brand_id").alias("brand_id"),
                    pl.col("i_brand").alias("brand"),
                    pl.col("ext_price"),
                ]
            )
            .sort(
                list(sort_by.keys()),
                descending=list(sort_by.values()),
                nulls_last=True,
            )
            .limit(limit)
        ),
        sort_by=list(sort_by.items()),
        limit=limit,
    )


def polars_impl_naive(run_config: RunConfig) -> QueryResult:
    """Query 52 (naive)."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=52,
        qualification=run_config.qualification,
    )

    year = params["year"]
    month = params["month"]
    manager_id = params["manager_id"]

    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    item = get_data(run_config.dataset_path, "item", run_config.suffix)
    return QueryResult(
        frame=(
            # SQL: FROM date_dim dt, store_sales, item
            # SQL: JOIN store_sales ON dt.d_date_sk = store_sales.ss_sold_date_sk
            date_dim.join(store_sales, left_on="d_date_sk", right_on="ss_sold_date_sk")
            # SQL: JOIN item ON store_sales.ss_item_sk = item.i_item_sk
            .join(item, left_on="ss_item_sk", right_on="i_item_sk")
            # SQL: WHERE item.i_manager_id={manager_id} AND dt.d_moy={month} AND dt.d_year={year}
            .filter(
                (pl.col("i_manager_id") == manager_id)
                & (pl.col("d_moy") == month)
                & (pl.col("d_year") == year)
            )
            # SQL: GROUP BY dt.d_year, item.i_brand, item.i_brand_id
            .group_by(["d_year", "i_brand", "i_brand_id"])
            # SQL: Sum(ss_ext_sales_price) AS ext_price
            .agg(pl.col("ss_ext_sales_price").sum().alias("ext_price"))
            # SQL: SELECT d_year, i_brand_id AS brand_id, i_brand AS brand, ext_price
            .select(
                [
                    pl.col("d_year"),
                    pl.col("i_brand_id").alias("brand_id"),
                    pl.col("i_brand").alias("brand"),
                    pl.col("ext_price"),
                ]
            )
            # SQL: ORDER BY d_year, ext_price DESC, brand_id
            .sort(["d_year", "ext_price", "brand_id"], descending=[False, True, False])
            # SQL: LIMIT 100
            .limit(100)
        ),
        sort_by=[("d_year", False), ("ext_price", True), ("brand_id", False)],
        limit=100,
    )
