# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 55."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.pdsds_parameters import load_parameters
from cudf_polars.experimental.benchmarks.utils import QueryResult, get_data

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 55."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=55,
        qualification=run_config.qualification,
    )

    year = params["year"]
    month = params["month"]
    manager_id = params["manager_id"]

    return f"""
    SELECT i_brand_id              brand_id,
                   i_brand                 brand,
                   Sum(ss_ext_sales_price) ext_price
    FROM   date_dim,
           store_sales,
           item
    WHERE  d_date_sk = ss_sold_date_sk
           AND ss_item_sk = i_item_sk
           AND i_manager_id = {manager_id}
           AND d_moy = {month}
           AND d_year = {year}
    GROUP  BY i_brand,
              i_brand_id
    ORDER  BY ext_price DESC,
              i_brand_id
    LIMIT 100;
    """


def polars_impl(run_config: RunConfig) -> QueryResult:
    """Query 55."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=55,
        qualification=run_config.qualification,
    )

    year = params["year"]
    month = params["month"]
    manager_id = params["manager_id"]

    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    item = get_data(run_config.dataset_path, "item", run_config.suffix)
    sort_by = {"ext_price": True, "brand_id": False}
    limit = 100

    # d_year not in GROUP BY so date filter can be a semi-join (no date columns needed).
    filtered_dates = date_dim.filter(
        (pl.col("d_moy") == month) & (pl.col("d_year") == year)
    ).select("d_date_sk")
    filtered_item = item.filter(pl.col("i_manager_id") == manager_id).select(
        ["i_item_sk", "i_brand", "i_brand_id"]
    )

    return QueryResult(
        frame=(
            store_sales.select(["ss_sold_date_sk", "ss_item_sk", "ss_ext_sales_price"])
            .join(filtered_dates, left_on="ss_sold_date_sk", right_on="d_date_sk", how="semi")
            .join(filtered_item, left_on="ss_item_sk", right_on="i_item_sk")
            .group_by(["i_brand", "i_brand_id"])
            .agg(pl.col("ss_ext_sales_price").sum().alias("ext_price"))
            .select(
                [
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
    """Query 55 (naive)."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=55,
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
            # SQL: FROM date_dim, store_sales, item
            # SQL: JOIN store_sales ON d_date_sk = ss_sold_date_sk
            date_dim.join(store_sales, left_on="d_date_sk", right_on="ss_sold_date_sk")
            # SQL: JOIN item ON ss_item_sk = i_item_sk
            .join(item, left_on="ss_item_sk", right_on="i_item_sk")
            # SQL: WHERE i_manager_id={manager_id} AND d_moy={month} AND d_year={year}
            .filter(
                (pl.col("i_manager_id") == manager_id)
                & (pl.col("d_moy") == month)
                & (pl.col("d_year") == year)
            )
            # SQL: GROUP BY i_brand, i_brand_id
            .group_by(["i_brand", "i_brand_id"])
            # SQL: Sum(ss_ext_sales_price) AS ext_price
            .agg(pl.col("ss_ext_sales_price").sum().alias("ext_price"))
            # SQL: SELECT i_brand_id AS brand_id, i_brand AS brand, ext_price
            .select(
                [
                    pl.col("i_brand_id").alias("brand_id"),
                    pl.col("i_brand").alias("brand"),
                    pl.col("ext_price"),
                ]
            )
            # SQL: ORDER BY ext_price DESC, i_brand_id
            .sort(["ext_price", "brand_id"], descending=[True, False])
            # SQL: LIMIT 100
            .limit(100)
        ),
        sort_by=[("ext_price", True), ("brand_id", False)],
        limit=100,
    )
