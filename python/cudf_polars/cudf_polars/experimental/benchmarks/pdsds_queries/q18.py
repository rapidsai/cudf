# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 18."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.utils import get_data

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 18."""
    return """
    SELECT i_item_id,
                   ca_country,
                   ca_state,
                   ca_county,
                   Avg(Cast(cs_quantity AS NUMERIC(12, 2)))      agg1,
                   Avg(Cast(cs_list_price AS NUMERIC(12, 2)))    agg2,
                   Avg(Cast(cs_coupon_amt AS NUMERIC(12, 2)))    agg3,
                   Avg(Cast(cs_sales_price AS NUMERIC(12, 2)))   agg4,
                   Avg(Cast(cs_net_profit AS NUMERIC(12, 2)))    agg5,
                   Avg(Cast(c_birth_year AS NUMERIC(12, 2)))     agg6,
                   Avg(Cast(cd1.cd_dep_count AS NUMERIC(12, 2))) agg7
    FROM   catalog_sales,
           customer_demographics cd1,
           customer_demographics cd2,
           customer,
           customer_address,
           date_dim,
           item
    WHERE  cs_sold_date_sk = d_date_sk
           AND cs_item_sk = i_item_sk
           AND cs_bill_cdemo_sk = cd1.cd_demo_sk
           AND cs_bill_customer_sk = c_customer_sk
           AND cd1.cd_gender = 'F'
           AND cd1.cd_education_status = 'Secondary'
           AND c_current_cdemo_sk = cd2.cd_demo_sk
           AND c_current_addr_sk = ca_address_sk
           AND c_birth_month IN ( 8, 4, 2, 5,
                                  11, 9 )
           AND d_year = 2001
           AND ca_state IN ( 'KS', 'IA', 'AL', 'UT',
                             'VA', 'NC', 'TX' )
    GROUP  BY rollup ( i_item_id, ca_country, ca_state, ca_county )
    ORDER  BY ca_country,
              ca_state,
              ca_county,
              i_item_id
    LIMIT 100;
    """


def level(  # noqa: D103
    base_query: pl.LazyFrame,
    agg_exprs: list[pl.Expr],
    null_sentinel: str,
    group_cols: list[str],
) -> pl.LazyFrame:
    if group_cols:
        lf = base_query.group_by(group_cols).agg(agg_exprs)
    else:
        lf = base_query.select(agg_exprs)
    missing = [
        c
        for c in ["i_item_id", "ca_country", "ca_state", "ca_county"]
        if c not in group_cols
    ]
    if missing:
        lf = lf.with_columns([pl.lit(null_sentinel).alias(c) for c in missing])
    return lf.select(
        [
            "i_item_id",
            "ca_country",
            "ca_state",
            "ca_county",
            "agg1",
            "agg2",
            "agg3",
            "agg4",
            "agg5",
            "agg6",
            "agg7",
        ]
    )


def polars_impl(run_config: RunConfig) -> pl.LazyFrame:
    """Query 18."""
    null_sentinel = "NULL"
    catalog_sales = get_data(
        run_config.dataset_path, "catalog_sales", run_config.suffix
    )
    customer_demographics_1 = get_data(
        run_config.dataset_path, "customer_demographics", run_config.suffix
    )
    customer_demographics_2 = get_data(
        run_config.dataset_path, "customer_demographics", run_config.suffix
    )
    customer = get_data(run_config.dataset_path, "customer", run_config.suffix)
    customer_address = get_data(
        run_config.dataset_path, "customer_address", run_config.suffix
    )
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    item = get_data(run_config.dataset_path, "item", run_config.suffix)

    base_query = (
        catalog_sales.join(date_dim, left_on="cs_sold_date_sk", right_on="d_date_sk")
        .join(item, left_on="cs_item_sk", right_on="i_item_sk")
        .join(
            customer_demographics_1,
            left_on="cs_bill_cdemo_sk",
            right_on="cd_demo_sk",
            suffix="_cd1",
        )
        .join(customer, left_on="cs_bill_customer_sk", right_on="c_customer_sk")
        .join(
            customer_demographics_2,
            left_on="c_current_cdemo_sk",
            right_on="cd_demo_sk",
            suffix="_cd2",
        )
        .join(customer_address, left_on="c_current_addr_sk", right_on="ca_address_sk")
        .filter(
            (pl.col("cd_gender") == "F")
            & (pl.col("cd_education_status") == "Secondary")
            & pl.col("c_birth_month").is_in([8, 4, 2, 5, 11, 9])
            & (pl.col("d_year") == 2001)
            & pl.col("ca_state").is_in(["KS", "IA", "AL", "UT", "VA", "NC", "TX"])
        )
    )

    agg_exprs = [
        pl.col("cs_quantity").mean().alias("agg1"),
        pl.col("cs_list_price").mean().alias("agg2"),
        pl.col("cs_coupon_amt").mean().alias("agg3"),
        pl.col("cs_sales_price").mean().alias("agg4"),
        pl.col("cs_net_profit").mean().alias("agg5"),
        pl.col("c_birth_year").mean().alias("agg6"),
        pl.col("cd_dep_count").mean().alias("agg7"),
    ]

    level1 = level(
        base_query,
        agg_exprs,
        null_sentinel,
        ["i_item_id", "ca_country", "ca_state", "ca_county"],
    )
    level2 = level(
        base_query, agg_exprs, null_sentinel, ["ca_country", "ca_state", "ca_county"]
    )
    level3 = level(base_query, agg_exprs, null_sentinel, ["ca_country", "ca_state"])
    level4 = level(base_query, agg_exprs, null_sentinel, ["ca_country"])
    level5 = level(base_query, agg_exprs, null_sentinel, [])

    return (
        pl.concat([level1, level2, level3, level4, level5])
        .filter(pl.col("i_item_id") != null_sentinel)
        .sort(["ca_country", "ca_state", "ca_county", "i_item_id"], nulls_last=True)
        .limit(100)
    )
