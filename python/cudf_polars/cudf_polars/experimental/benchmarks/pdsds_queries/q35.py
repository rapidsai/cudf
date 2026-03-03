# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 35."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.pdsds_parameters import load_parameters
from cudf_polars.experimental.benchmarks.utils import get_data

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 35."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=35,
        qualification=run_config.qualification,
    )

    year = params["year"]
    aggone = params["aggone"]
    aggtwo = params["aggtwo"]
    aggthree = params["aggthree"]

    return f"""
    SELECT ca_state,
                   cd_gender,
                   cd_marital_status,
                   cd_dep_count,
                   Count(*) cnt1,
                   {aggone}(cd_dep_count),
                   {aggtwo}(cd_dep_count),
                   {aggthree}(cd_dep_count),
                   cd_dep_employed_count,
                   Count(*) cnt2,
                   {aggone}(cd_dep_employed_count),
                   {aggtwo}(cd_dep_employed_count),
                   {aggthree}(cd_dep_employed_count),
                   cd_dep_college_count,
                   Count(*) cnt3,
                   {aggone}(cd_dep_college_count),
                   {aggtwo}(cd_dep_college_count),
                   {aggthree}(cd_dep_college_count)
    FROM   customer c,
           customer_address ca,
           customer_demographics
    WHERE  c.c_current_addr_sk = ca.ca_address_sk
           AND cd_demo_sk = c.c_current_cdemo_sk
           AND EXISTS (SELECT *
                       FROM   store_sales,
                              date_dim
                       WHERE  c.c_customer_sk = ss_customer_sk
                              AND ss_sold_date_sk = d_date_sk
                              AND d_year = {year}
                              AND d_qoy < 4)
           AND ( EXISTS (SELECT *
                         FROM   web_sales,
                                date_dim
                         WHERE  c.c_customer_sk = ws_bill_customer_sk
                                AND ws_sold_date_sk = d_date_sk
                                AND d_year = {year}
                                AND d_qoy < 4)
                  OR EXISTS (SELECT *
                             FROM   catalog_sales,
                                    date_dim
                             WHERE  c.c_customer_sk = cs_ship_customer_sk
                                    AND cs_sold_date_sk = d_date_sk
                                    AND d_year = {year}
                                    AND d_qoy < 4) )
    GROUP  BY ca_state,
              cd_gender,
              cd_marital_status,
              cd_dep_count,
              cd_dep_employed_count,
              cd_dep_college_count
    ORDER  BY ca_state,
              cd_gender,
              cd_marital_status,
              cd_dep_count,
              cd_dep_employed_count,
              cd_dep_college_count
    LIMIT 100;
    """


def _get_agg_expr(col_name: str, agg_func: str, alias: str) -> pl.Expr:
    col = pl.col(col_name)
    if agg_func == "sum":
        return col.sum().alias(alias)
    elif agg_func == "min":
        return col.min().alias(alias)
    elif agg_func == "max":
        return col.max().alias(alias)
    elif agg_func == "avg":
        return col.mean().alias(alias)
    elif agg_func == "stddev_samp":
        return col.std().alias(alias)
    else:
        msg = f"Unknown aggregation function: {agg_func}"
        raise ValueError(msg)


def polars_impl(run_config: RunConfig) -> pl.LazyFrame:
    """Query 35."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=35,
        qualification=run_config.qualification,
    )

    year = params["year"]
    aggone = params["aggone"]
    aggtwo = params["aggtwo"]
    aggthree = params["aggthree"]

    # Load tables
    customer = get_data(run_config.dataset_path, "customer", run_config.suffix)
    customer_address = get_data(
        run_config.dataset_path, "customer_address", run_config.suffix
    )
    customer_demographics = get_data(
        run_config.dataset_path, "customer_demographics", run_config.suffix
    )
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    web_sales = get_data(run_config.dataset_path, "web_sales", run_config.suffix)
    catalog_sales = get_data(
        run_config.dataset_path, "catalog_sales", run_config.suffix
    )
    # Main customer data
    main_customers = customer.join(
        customer_address, left_on="c_current_addr_sk", right_on="ca_address_sk"
    ).join(customer_demographics, left_on="c_current_cdemo_sk", right_on="cd_demo_sk")
    # Store sales existence check
    ss_customers = (
        store_sales.join(date_dim, left_on="ss_sold_date_sk", right_on="d_date_sk")
        .filter((pl.col("d_year") == year) & (pl.col("d_qoy") < 4))
        .select("ss_customer_sk")
        .unique()
    )
    # Web sales existence check
    ws_customers = (
        web_sales.join(date_dim, left_on="ws_sold_date_sk", right_on="d_date_sk")
        .filter((pl.col("d_year") == year) & (pl.col("d_qoy") < 4))
        .select("ws_bill_customer_sk")
        .unique()
    )
    # Catalog sales existence check
    cs_customers = (
        catalog_sales.join(date_dim, left_on="cs_sold_date_sk", right_on="d_date_sk")
        .filter((pl.col("d_year") == year) & (pl.col("d_qoy") < 4))
        .select("cs_ship_customer_sk")
        .unique()
    )
    # Apply EXISTS conditions
    return (
        main_customers.join(
            ss_customers, left_on="c_customer_sk", right_on="ss_customer_sk"
        )
        .join(
            pl.concat(
                [
                    ws_customers.rename({"ws_bill_customer_sk": "customer_sk"}),
                    cs_customers.rename({"cs_ship_customer_sk": "customer_sk"}),
                ]
            ).unique(),
            left_on="c_customer_sk",
            right_on="customer_sk",
        )
        .group_by(
            [
                "ca_state",
                "cd_gender",
                "cd_marital_status",
                "cd_dep_count",
                "cd_dep_employed_count",
                "cd_dep_college_count",
            ]
        )
        .agg(
            [
                pl.len().alias("cnt1"),
                _get_agg_expr("cd_dep_count", aggone, f"{aggone}(cd_dep_count)"),
                _get_agg_expr("cd_dep_count", aggtwo, f"{aggtwo}(cd_dep_count)"),
                _get_agg_expr("cd_dep_count", aggthree, f"{aggthree}(cd_dep_count)"),
                pl.len().alias("cnt2"),
                _get_agg_expr(
                    "cd_dep_employed_count", aggone, f"{aggone}(cd_dep_employed_count)"
                ),
                _get_agg_expr(
                    "cd_dep_employed_count", aggtwo, f"{aggtwo}(cd_dep_employed_count)"
                ),
                _get_agg_expr(
                    "cd_dep_employed_count",
                    aggthree,
                    f"{aggthree}(cd_dep_employed_count)",
                ),
                pl.len().alias("cnt3"),
                _get_agg_expr(
                    "cd_dep_college_count", aggone, f"{aggone}(cd_dep_college_count)"
                ),
                _get_agg_expr(
                    "cd_dep_college_count", aggtwo, f"{aggtwo}(cd_dep_college_count)"
                ),
                _get_agg_expr(
                    "cd_dep_college_count",
                    aggthree,
                    f"{aggthree}(cd_dep_college_count)",
                ),
            ]
        )
        .select(
            [
                "ca_state",
                "cd_gender",
                "cd_marital_status",
                "cd_dep_count",
                "cnt1",
                f"{aggone}(cd_dep_count)",
                f"{aggtwo}(cd_dep_count)",
                f"{aggthree}(cd_dep_count)",
                "cd_dep_employed_count",
                "cnt2",
                f"{aggone}(cd_dep_employed_count)",
                f"{aggtwo}(cd_dep_employed_count)",
                f"{aggthree}(cd_dep_employed_count)",
                "cd_dep_college_count",
                "cnt3",
                f"{aggone}(cd_dep_college_count)",
                f"{aggtwo}(cd_dep_college_count)",
                f"{aggthree}(cd_dep_college_count)",
            ]
        )
        .sort(
            [
                "ca_state",
                "cd_gender",
                "cd_marital_status",
                "cd_dep_count",
                "cd_dep_employed_count",
                "cd_dep_college_count",
            ],
            nulls_last=True,
        )
        .limit(100)
    )
