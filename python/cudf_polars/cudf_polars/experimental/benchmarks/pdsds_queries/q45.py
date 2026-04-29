# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 45."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.pdsds_parameters import load_parameters
from cudf_polars.experimental.benchmarks.utils import QueryResult, get_data

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 45."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=45,
        qualification=run_config.qualification,
    )

    year = params["year"]
    qoy = params["qoy"]
    zip_codes = params["zip_codes"]
    item_sks = params["item_sks"]

    zip_codes_str = ", ".join(f"'{z}'" for z in zip_codes)
    item_sks_str = ", ".join(str(i) for i in item_sks)

    return f"""
    SELECT ca_zip,
                   ca_state,
                   Sum(ws_sales_price)
    FROM   web_sales,
           customer,
           customer_address,
           date_dim,
           item
    WHERE  ws_bill_customer_sk = c_customer_sk
           AND c_current_addr_sk = ca_address_sk
           AND ws_item_sk = i_item_sk
           AND ( Substr(ca_zip, 1, 5) IN ( {zip_codes_str} )
                  OR i_item_id IN (SELECT i_item_id
                                   FROM   item
                                   WHERE  i_item_sk IN ( {item_sks_str} )) )
           AND ws_sold_date_sk = d_date_sk
           AND d_qoy = {qoy}
           AND d_year = {year}
    GROUP  BY ca_zip,
              ca_state
    ORDER  BY ca_zip,
              ca_state
    LIMIT 100;
    """


def polars_impl(run_config: RunConfig) -> QueryResult:
    """Query 45."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=45,
        qualification=run_config.qualification,
    )

    year = params["year"]
    qoy = params["qoy"]
    zip_codes = params["zip_codes"]
    item_sks = params["item_sks"]

    # Load tables
    web_sales = get_data(run_config.dataset_path, "web_sales", run_config.suffix)
    customer = get_data(run_config.dataset_path, "customer", run_config.suffix)
    customer_address = get_data(
        run_config.dataset_path,
        "customer_address",
        run_config.suffix,
    )
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    item = get_data(run_config.dataset_path, "item", run_config.suffix)

    # Subquery: filter item table to just those i_item_id matching i_item_sk
    filtered_items = (
        item.filter(pl.col("i_item_sk").is_in(item_sks)).select("i_item_id").unique()
    )

    # Perform all joins first and filter for date
    joined = (
        web_sales.join(
            customer, left_on="ws_bill_customer_sk", right_on="c_customer_sk"
        )
        .join(customer_address, left_on="c_current_addr_sk", right_on="ca_address_sk")
        .join(item, left_on="ws_item_sk", right_on="i_item_sk")
        .join(date_dim, left_on="ws_sold_date_sk", right_on="d_date_sk")
        .filter((pl.col("d_qoy") == qoy) & (pl.col("d_year") == year))
        # Extract first 5 characters of ZIP code
        .with_columns([pl.col("ca_zip").str.slice(0, 5).alias("zip_prefix")])
    )

    # First condition: zip code prefix in target list
    zip_match = joined.filter(pl.col("zip_prefix").is_in(zip_codes))

    # Second condition: item ID in filtered subquery result
    item_match = joined.join(
        filtered_items, left_on="i_item_id", right_on="i_item_id", how="semi"
    )

    sort_by = {"ca_zip": False, "ca_state": False}
    limit = 100
    return QueryResult(
        frame=(
            pl.concat([zip_match, item_match])
            .group_by(["ca_zip", "ca_state"])
            .agg(
                [
                    pl.col("ws_sales_price").sum().alias("sum(ws_sales_price)"),
                ]
            )
            .sort(sort_by.keys(), nulls_last=True)
            .select(["ca_zip", "ca_state", "sum(ws_sales_price)"])
            .limit(limit)
        ),
        sort_by=list(sort_by.items()),
        limit=limit,
    )


def polars_impl_naive(run_config: RunConfig) -> QueryResult:
    """Query 45 (naive)."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=45,
        qualification=run_config.qualification,
    )

    year = params["year"]
    qoy = params["qoy"]
    zip_codes = params["zip_codes"]
    item_sks = params["item_sks"]

    web_sales = get_data(run_config.dataset_path, "web_sales", run_config.suffix)
    customer = get_data(run_config.dataset_path, "customer", run_config.suffix)
    customer_address = get_data(
        run_config.dataset_path,
        "customer_address",
        run_config.suffix,
    )
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    item = get_data(run_config.dataset_path, "item", run_config.suffix)

    # SQL: subquery — i_item_id WHERE i_item_sk IN ({item_sks})
    item_ids = (
        item.filter(pl.col("i_item_sk").is_in(item_sks)).select("i_item_id").unique()
    )

    # SQL: FROM web_sales, customer, customer_address, date_dim, item
    joined = (
        web_sales.join(
            customer, left_on="ws_bill_customer_sk", right_on="c_customer_sk"
        )
        .join(customer_address, left_on="c_current_addr_sk", right_on="ca_address_sk")
        # SQL: JOIN date_dim ON ws_sold_date_sk = d_date_sk (before item, matching SQL FROM order)
        .join(date_dim, left_on="ws_sold_date_sk", right_on="d_date_sk")
        .join(item, left_on="ws_item_sk", right_on="i_item_sk")
        .with_columns([pl.col("ca_zip").str.slice(0, 5).alias("zip_prefix")])
    )

    # SQL: WHERE (Substr(ca_zip,1,5) IN ({zip_codes}) OR i_item_id IN (subquery))
    # SQL:   AND d_qoy={qoy} AND d_year={year}
    # These are the two branches of the OR, each filtered after all joins:
    zip_match = joined.filter(
        (pl.col("d_qoy") == qoy)
        & (pl.col("d_year") == year)
        & pl.col("zip_prefix").is_in(zip_codes)
    )
    # SQL: WHERE i_item_id IN (subquery) AND d_qoy={qoy} AND d_year={year}
    item_match = joined.filter(
        (pl.col("d_qoy") == qoy) & (pl.col("d_year") == year)
    ).join(item_ids, on="i_item_id", how="semi")

    sort_by = {"ca_zip": False, "ca_state": False}
    limit = 100
    return QueryResult(
        frame=(
            # SQL: UNION (both WHERE conditions) — OR in original SQL
            pl.concat([zip_match, item_match])
            # SQL: GROUP BY ca_zip, ca_state
            .group_by(["ca_zip", "ca_state"])
            # SQL: Sum(ws_sales_price) AS sum(ws_sales_price)
            .agg(pl.col("ws_sales_price").sum().alias("sum(ws_sales_price)"))
            # SQL: ORDER BY ca_zip, ca_state
            .sort(sort_by.keys(), nulls_last=True)
            # SQL: SELECT ca_zip, ca_state, sum(ws_sales_price)
            .select(["ca_zip", "ca_state", "sum(ws_sales_price)"])
            # SQL: LIMIT 100
            .limit(limit)
        ),
        sort_by=list(sort_by.items()),
        limit=limit,
    )
