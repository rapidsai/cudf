# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 38."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.pdsds_parameters import load_parameters
from cudf_polars.experimental.benchmarks.utils import QueryResult, get_data

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 38."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=38,
        qualification=run_config.qualification,
    )

    dms = params["dms"]

    return f"""
    SELECT Count(*)
    FROM   (SELECT DISTINCT c_last_name,
                            c_first_name,
                            d_date
            FROM   store_sales,
                   date_dim,
                   customer
            WHERE  store_sales.ss_sold_date_sk = date_dim.d_date_sk
                   AND store_sales.ss_customer_sk = customer.c_customer_sk
                   AND d_month_seq BETWEEN {dms} AND {dms} + 11
            INTERSECT
            SELECT DISTINCT c_last_name,
                            c_first_name,
                            d_date
            FROM   catalog_sales,
                   date_dim,
                   customer
            WHERE  catalog_sales.cs_sold_date_sk = date_dim.d_date_sk
                   AND catalog_sales.cs_bill_customer_sk = customer.c_customer_sk
                   AND d_month_seq BETWEEN {dms} AND {dms} + 11
            INTERSECT
            SELECT DISTINCT c_last_name,
                            c_first_name,
                            d_date
            FROM   web_sales,
                   date_dim,
                   customer
            WHERE  web_sales.ws_sold_date_sk = date_dim.d_date_sk
                   AND web_sales.ws_bill_customer_sk = customer.c_customer_sk
                   AND d_month_seq BETWEEN {dms} AND {dms} + 11) hot_cust
    LIMIT 100;
    """


def polars_impl(run_config: RunConfig) -> QueryResult:
    """Query 38."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=38,
        qualification=run_config.qualification,
    )

    dms = params["dms"]

    # Load tables
    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    catalog_sales = get_data(
        run_config.dataset_path, "catalog_sales", run_config.suffix
    )
    web_sales = get_data(run_config.dataset_path, "web_sales", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    customer = get_data(run_config.dataset_path, "customer", run_config.suffix)
    # Filter date_dim for the specified month sequence range
    date_filter = date_dim.filter(pl.col("d_month_seq").is_between(dms, dms + 11))
    store_customers = (
        store_sales.join(date_filter, left_on="ss_sold_date_sk", right_on="d_date_sk")
        .join(customer, left_on="ss_customer_sk", right_on="c_customer_sk")
        .select(["c_last_name", "c_first_name", "d_date"])
        .unique()
    )
    catalog_customers = (
        catalog_sales.join(date_filter, left_on="cs_sold_date_sk", right_on="d_date_sk")
        .join(customer, left_on="cs_bill_customer_sk", right_on="c_customer_sk")
        .select(["c_last_name", "c_first_name", "d_date"])
        .unique()
    )
    web_customers = (
        web_sales.join(date_filter, left_on="ws_sold_date_sk", right_on="d_date_sk")
        .join(customer, left_on="ws_bill_customer_sk", right_on="c_customer_sk")
        .select(["c_last_name", "c_first_name", "d_date"])
        .unique()
    )
    # TODO: benchmark alternative INTERSECT strategies (source-tagging vs sequential join)
    intersect_first = store_customers.join(
        catalog_customers,
        on=["c_last_name", "c_first_name", "d_date"],
        nulls_equal=True,
    ).unique()
    intersect_final = intersect_first.join(
        web_customers,
        on=["c_last_name", "c_first_name", "d_date"],
        nulls_equal=True,
    ).unique()
    limit = 100
    return QueryResult(
        frame=(
            intersect_final.select(
                [pl.len().cast(pl.Int64).alias("count_star()")]
            ).limit(limit)
        ),
        sort_by=[],
        limit=limit,
    )


def polars_impl_naive(run_config: RunConfig) -> QueryResult:
    """Query 38 (naive)."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=38,
        qualification=run_config.qualification,
    )

    dms = params["dms"]

    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    catalog_sales = get_data(
        run_config.dataset_path, "catalog_sales", run_config.suffix
    )
    web_sales = get_data(run_config.dataset_path, "web_sales", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    customer = get_data(run_config.dataset_path, "customer", run_config.suffix)

    # SQL: store customers subquery — DISTINCT c_last_name, c_first_name, d_date FROM store_sales, date_dim, customer WHERE ss_sold_date_sk=d_date_sk AND ss_customer_sk=c_customer_sk AND d_month_seq BETWEEN {dms} AND {dms}+11
    store_customers = (
        store_sales.join(date_dim, left_on="ss_sold_date_sk", right_on="d_date_sk")
        .join(customer, left_on="ss_customer_sk", right_on="c_customer_sk")
        .filter(pl.col("d_month_seq").is_between(dms, dms + 11))
        .select(["c_last_name", "c_first_name", "d_date"])
        .unique()
    )

    # SQL: catalog customers subquery — same for catalog_sales with cs_bill_customer_sk
    catalog_customers = (
        catalog_sales.join(date_dim, left_on="cs_sold_date_sk", right_on="d_date_sk")
        .join(customer, left_on="cs_bill_customer_sk", right_on="c_customer_sk")
        .filter(pl.col("d_month_seq").is_between(dms, dms + 11))
        .select(["c_last_name", "c_first_name", "d_date"])
        .unique()
    )

    # SQL: web customers subquery — same for web_sales with ws_bill_customer_sk
    web_customers = (
        web_sales.join(date_dim, left_on="ws_sold_date_sk", right_on="d_date_sk")
        .join(customer, left_on="ws_bill_customer_sk", right_on="c_customer_sk")
        .filter(pl.col("d_month_seq").is_between(dms, dms + 11))
        .select(["c_last_name", "c_first_name", "d_date"])
        .unique()
    )

    # SQL: INTERSECT of store/catalog/web customer name+date sets
    intersect_first = store_customers.join(
        catalog_customers,
        on=["c_last_name", "c_first_name", "d_date"],
        nulls_equal=True,
    ).unique()
    # SQL: second INTERSECT with web customers
    intersect_final = intersect_first.join(
        web_customers,
        on=["c_last_name", "c_first_name", "d_date"],
        nulls_equal=True,
    ).unique()

    return QueryResult(
        frame=(
            # SQL: SELECT Count(*) FROM (INTERSECT subquery) hot_cust
            intersect_final.select(
                [pl.len().cast(pl.Int64).alias("count_star()")]
                # SQL: LIMIT 100
            ).limit(100)
        ),
        sort_by=[],
        limit=100,
    )
