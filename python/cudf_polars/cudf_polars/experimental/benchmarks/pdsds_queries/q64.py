# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 64."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.utils import get_data

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 64."""
    return """
    WITH cs_ui
         AS (SELECT cs_item_sk,
                    Sum(cs_ext_list_price) AS sale,
                    Sum(cr_refunded_cash + cr_reversed_charge
                        + cr_store_credit) AS refund
             FROM   catalog_sales,
                    catalog_returns
             WHERE  cs_item_sk = cr_item_sk
                    AND cs_order_number = cr_order_number
             GROUP  BY cs_item_sk
             HAVING Sum(cs_ext_list_price) > 2 * Sum(
                    cr_refunded_cash + cr_reversed_charge
                    + cr_store_credit)),
         cross_sales
         AS (SELECT i_product_name         product_name,
                    i_item_sk              item_sk,
                    s_store_name           store_name,
                    s_zip                  store_zip,
                    ad1.ca_street_number   b_street_number,
                    ad1.ca_street_name     b_streen_name,
                    ad1.ca_city            b_city,
                    ad1.ca_zip             b_zip,
                    ad2.ca_street_number   c_street_number,
                    ad2.ca_street_name     c_street_name,
                    ad2.ca_city            c_city,
                    ad2.ca_zip             c_zip,
                    d1.d_year              AS syear,
                    d2.d_year              AS fsyear,
                    d3.d_year              s2year,
                    Count(*)               cnt,
                    Sum(ss_wholesale_cost) s1,
                    Sum(ss_list_price)     s2,
                    Sum(ss_coupon_amt)     s3
             FROM   store_sales,
                    store_returns,
                    cs_ui,
                    date_dim d1,
                    date_dim d2,
                    date_dim d3,
                    store,
                    customer,
                    customer_demographics cd1,
                    customer_demographics cd2,
                    promotion,
                    household_demographics hd1,
                    household_demographics hd2,
                    customer_address ad1,
                    customer_address ad2,
                    income_band ib1,
                    income_band ib2,
                    item
             WHERE  ss_store_sk = s_store_sk
                    AND ss_sold_date_sk = d1.d_date_sk
                    AND ss_customer_sk = c_customer_sk
                    AND ss_cdemo_sk = cd1.cd_demo_sk
                    AND ss_hdemo_sk = hd1.hd_demo_sk
                    AND ss_addr_sk = ad1.ca_address_sk
                    AND ss_item_sk = i_item_sk
                    AND ss_item_sk = sr_item_sk
                    AND ss_ticket_number = sr_ticket_number
                    AND ss_item_sk = cs_ui.cs_item_sk
                    AND c_current_cdemo_sk = cd2.cd_demo_sk
                    AND c_current_hdemo_sk = hd2.hd_demo_sk
                    AND c_current_addr_sk = ad2.ca_address_sk
                    AND c_first_sales_date_sk = d2.d_date_sk
                    AND c_first_shipto_date_sk = d3.d_date_sk
                    AND ss_promo_sk = p_promo_sk
                    AND hd1.hd_income_band_sk = ib1.ib_income_band_sk
                    AND hd2.hd_income_band_sk = ib2.ib_income_band_sk
                    AND cd1.cd_marital_status <> cd2.cd_marital_status
                    AND i_color IN ( 'cyan', 'peach', 'blush', 'frosted',
                                     'powder', 'orange' )
                    AND i_current_price BETWEEN 58 AND 58 + 10
                    AND i_current_price BETWEEN 58 + 1 AND 58 + 15
             GROUP  BY i_product_name,
                       i_item_sk,
                       s_store_name,
                       s_zip,
                       ad1.ca_street_number,
                       ad1.ca_street_name,
                       ad1.ca_city,
                       ad1.ca_zip,
                       ad2.ca_street_number,
                       ad2.ca_street_name,
                       ad2.ca_city,
                       ad2.ca_zip,
                       d1.d_year,
                       d2.d_year,
                       d3.d_year)
    SELECT cs1.product_name,
           cs1.store_name,
           cs1.store_zip,
           cs1.b_street_number,
           cs1.b_streen_name,
           cs1.b_city,
           cs1.b_zip,
           cs1.c_street_number,
           cs1.c_street_name,
           cs1.c_city,
           cs1.c_zip,
           cs1.syear,
           cs1.cnt,
           cs1.s1,
           cs1.s2,
           cs1.s3,
           cs2.s1,
           cs2.s2,
           cs2.s3,
           cs2.syear,
           cs2.cnt
    FROM   cross_sales cs1,
           cross_sales cs2
    WHERE  cs1.item_sk = cs2.item_sk
           AND cs1.syear = 2001
           AND cs2.syear = 2001 + 1
           AND cs2.cnt <= cs1.cnt
           AND cs1.store_name = cs2.store_name
           AND cs1.store_zip = cs2.store_zip
    ORDER  BY cs1.product_name,
              cs1.store_name,
              cs2.cnt,
              cs1.s1,
              cs2.s1;
    """


def polars_impl(run_config: RunConfig) -> pl.LazyFrame:
    """Query 64."""
    catalog_sales = get_data(
        run_config.dataset_path, "catalog_sales", run_config.suffix
    )
    catalog_returns = get_data(
        run_config.dataset_path, "catalog_returns", run_config.suffix
    )
    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    store_returns = get_data(
        run_config.dataset_path, "store_returns", run_config.suffix
    )
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    store = get_data(run_config.dataset_path, "store", run_config.suffix)
    customer = get_data(run_config.dataset_path, "customer", run_config.suffix)
    customer_demographics = get_data(
        run_config.dataset_path, "customer_demographics", run_config.suffix
    )
    customer_address = get_data(
        run_config.dataset_path, "customer_address", run_config.suffix
    )
    item = get_data(run_config.dataset_path, "item", run_config.suffix)

    cs_ui_items = (
        catalog_sales.select(["cs_item_sk", "cs_order_number", "cs_ext_list_price"])
        .join(
            catalog_returns.select(
                [
                    "cr_item_sk",
                    "cr_order_number",
                    "cr_refunded_cash",
                    "cr_reversed_charge",
                    "cr_store_credit",
                ]
            ),
            left_on=["cs_item_sk", "cs_order_number"],
            right_on=["cr_item_sk", "cr_order_number"],
            how="inner",
        )
        .group_by("cs_item_sk")
        .agg(
            [
                pl.col("cs_ext_list_price").sum().alias("sale"),
                (
                    pl.col("cr_refunded_cash")
                    + pl.col("cr_reversed_charge")
                    + pl.col("cr_store_credit")
                )
                .sum()
                .alias("refund"),
            ]
        )
        .filter(pl.col("sale") > 2 * pl.col("refund"))
        .select("cs_item_sk")
    )

    filtered_items = item.filter(
        pl.col("i_color").is_in(
            ["cyan", "peach", "blush", "frosted", "powder", "orange"]
        )
        & pl.col("i_current_price").is_between(58, 68)
        & pl.col("i_current_price").is_between(59, 73)
    ).select(["i_item_sk", "i_product_name"])

    base = (
        store_sales.join(
            filtered_items, left_on="ss_item_sk", right_on="i_item_sk", how="inner"
        )
        .join(cs_ui_items, left_on="ss_item_sk", right_on="cs_item_sk", how="semi")
        .join(
            store_returns.select(["sr_item_sk", "sr_ticket_number"]),
            left_on=["ss_item_sk", "ss_ticket_number"],
            right_on=["sr_item_sk", "sr_ticket_number"],
            how="inner",
        )
        .select(
            [
                "ss_item_sk",
                "ss_store_sk",
                "ss_sold_date_sk",
                "ss_customer_sk",
                "ss_cdemo_sk",
                "ss_addr_sk",
                "ss_wholesale_cost",
                "ss_list_price",
                "ss_coupon_amt",
                pl.col("i_product_name"),
            ]
        )
    )

    base = (
        base.join(
            store.select(["s_store_sk", "s_store_name", "s_zip"]),
            left_on="ss_store_sk",
            right_on="s_store_sk",
            how="inner",
        )
        .join(
            date_dim.select(["d_date_sk", "d_year"]).rename({"d_year": "syear"}),
            left_on="ss_sold_date_sk",
            right_on="d_date_sk",
            how="inner",
        )
        .join(
            customer.select(
                [
                    "c_customer_sk",
                    "c_current_cdemo_sk",
                    "c_current_addr_sk",
                    "c_first_sales_date_sk",
                    "c_first_shipto_date_sk",
                ]
            ),
            left_on="ss_customer_sk",
            right_on="c_customer_sk",
            how="inner",
        )
        .join(
            customer_demographics.select(["cd_demo_sk", "cd_marital_status"]).rename(
                {"cd_marital_status": "cd1_marital"}
            ),
            left_on="ss_cdemo_sk",
            right_on="cd_demo_sk",
            how="inner",
        )
        .join(
            customer_demographics.select(["cd_demo_sk", "cd_marital_status"]).rename(
                {"cd_demo_sk": "cd2_demo_sk", "cd_marital_status": "cd2_marital"}
            ),
            left_on="c_current_cdemo_sk",
            right_on="cd2_demo_sk",
            how="inner",
        )
        .filter(pl.col("cd1_marital") != pl.col("cd2_marital"))
        .join(
            customer_address.select(
                [
                    "ca_address_sk",
                    "ca_street_number",
                    "ca_street_name",
                    "ca_city",
                    "ca_zip",
                ]
            ).rename(
                {
                    "ca_address_sk": "b_addr_sk",
                    "ca_street_number": "b_street_number",
                    "ca_street_name": "b_streen_name",
                    "ca_city": "b_city",
                    "ca_zip": "b_zip",
                }
            ),
            left_on="ss_addr_sk",
            right_on="b_addr_sk",
            how="inner",
        )
        .join(
            customer_address.select(
                [
                    "ca_address_sk",
                    "ca_street_number",
                    "ca_street_name",
                    "ca_city",
                    "ca_zip",
                ]
            ).rename(
                {
                    "ca_address_sk": "c_addr_sk",
                    "ca_street_number": "c_street_number",
                    "ca_street_name": "c_street_name",
                    "ca_city": "c_city",
                    "ca_zip": "c_zip",
                }
            ),
            left_on="c_current_addr_sk",
            right_on="c_addr_sk",
            how="inner",
        )
        .join(
            date_dim.select(["d_date_sk", "d_year"]).rename(
                {"d_date_sk": "d2_date_sk", "d_year": "fsyear"}
            ),
            left_on="c_first_sales_date_sk",
            right_on="d2_date_sk",
            how="inner",
        )
        .join(
            date_dim.select(["d_date_sk", "d_year"]).rename(
                {"d_date_sk": "d3_date_sk", "d_year": "s2year"}
            ),
            left_on="c_first_shipto_date_sk",
            right_on="d3_date_sk",
            how="inner",
        )
    )

    cross_sales = (
        base.group_by(
            [
                "i_product_name",
                "ss_item_sk",
                "s_store_name",
                "s_zip",
                "b_street_number",
                "b_streen_name",
                "b_city",
                "b_zip",
                "c_street_number",
                "c_street_name",
                "c_city",
                "c_zip",
                "syear",
                "fsyear",
                "s2year",
            ]
        )
        .agg(
            [
                pl.len().cast(pl.Int64).alias("cnt"),
                pl.col("ss_wholesale_cost").sum().alias("s1"),
                pl.col("ss_list_price").sum().alias("s2"),
                pl.col("ss_coupon_amt").sum().alias("s3"),
            ]
        )
        .with_columns(
            [
                pl.col("i_product_name").alias("product_name"),
                pl.col("ss_item_sk").alias("item_sk"),
                pl.col("s_store_name").alias("store_name"),
                pl.col("s_zip").alias("store_zip"),
            ]
        )
    )

    return (
        cross_sales.join(
            cross_sales,
            left_on=["item_sk", "store_name", "store_zip"],
            right_on=["item_sk", "store_name", "store_zip"],
            suffix="_cs2",
        )
        .filter(
            (pl.col("syear") == 2001)
            & (pl.col("syear_cs2") == 2002)
            & (pl.col("cnt_cs2") <= pl.col("cnt"))
        )
        .select(
            [
                "product_name",
                "store_name",
                "store_zip",
                "b_street_number",
                "b_streen_name",
                "b_city",
                "b_zip",
                "c_street_number",
                "c_street_name",
                "c_city",
                "c_zip",
                "syear",
                "cnt",
                "s1",
                "s2",
                "s3",
                pl.col("s1_cs2").alias("s1_1"),
                pl.col("s2_cs2").alias("s2_1"),
                pl.col("s3_cs2").alias("s3_1"),
                pl.col("syear_cs2").alias("syear_1"),
                pl.col("cnt_cs2").cast(pl.Int64).alias("cnt_1"),
            ]
        )
        .sort(
            ["product_name", "store_name", "cnt_1", "s1", "s1_1"],
            nulls_last=True,
            descending=[False, False, False, False, False],
        )
        .limit(100)
    )
