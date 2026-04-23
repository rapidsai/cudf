# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 23."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.pdsds_parameters import load_parameters
from cudf_polars.experimental.benchmarks.utils import QueryResult, get_data

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 23."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=23,
        qualification=run_config.qualification,
    )

    year = params["year"]
    month = params["month"]

    return f"""
    WITH frequent_ss_items AS
      (SELECT itemdesc,
              i_item_sk item_sk,
              d_date solddate,
              count(*) cnt
       FROM store_sales,
            date_dim,
            (SELECT SUBSTRING(i_item_desc, 1, 30) itemdesc, *
             FROM item) sq1
       WHERE ss_sold_date_sk = d_date_sk
         AND ss_item_sk = i_item_sk
         AND d_year IN ({year}, {year}+1, {year}+2, {year}+3)
       GROUP BY itemdesc, i_item_sk, d_date
       HAVING count(*) > 4),
    max_store_sales AS
      (SELECT max(csales) tpcds_cmax
       FROM (SELECT c_customer_sk,
                    sum(ss_quantity*ss_sales_price) csales
             FROM store_sales, customer, date_dim
             WHERE ss_customer_sk = c_customer_sk
               AND ss_sold_date_sk = d_date_sk
               AND d_year IN ({year}, {year}+1, {year}+2, {year}+3)
             GROUP BY c_customer_sk) sq2),
    best_ss_customer AS
      (SELECT c_customer_sk,
              sum(ss_quantity*ss_sales_price) ssales
       FROM store_sales, customer, max_store_sales
       WHERE ss_customer_sk = c_customer_sk
       GROUP BY c_customer_sk
       HAVING sum(ss_quantity*ss_sales_price) > (95/100.0) * max(tpcds_cmax))
    SELECT c_last_name, c_first_name, sales
    FROM (SELECT c_last_name, c_first_name,
                 sum(cs_quantity*cs_list_price) sales
          FROM catalog_sales, customer, date_dim, frequent_ss_items, best_ss_customer
          WHERE d_year = {year}
            AND d_moy = {month}
            AND cs_sold_date_sk = d_date_sk
            AND cs_item_sk = item_sk
            AND cs_bill_customer_sk = best_ss_customer.c_customer_sk
            AND cs_bill_customer_sk = customer.c_customer_sk
          GROUP BY c_last_name, c_first_name
          UNION ALL
          SELECT c_last_name, c_first_name,
                 sum(ws_quantity*ws_list_price) sales
          FROM web_sales, customer, date_dim, frequent_ss_items, best_ss_customer
          WHERE d_year = {year}
            AND d_moy = {month}
            AND ws_sold_date_sk = d_date_sk
            AND ws_item_sk = item_sk
            AND ws_bill_customer_sk = best_ss_customer.c_customer_sk
            AND ws_bill_customer_sk = customer.c_customer_sk
          GROUP BY c_last_name, c_first_name) sq3
    ORDER BY c_last_name NULLS FIRST, c_first_name NULLS FIRST, sales NULLS FIRST
    LIMIT 100;
    """


def polars_impl(run_config: RunConfig) -> QueryResult:
    """Query 23."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=23,
        qualification=run_config.qualification,
    )

    year = params["year"]
    month = params["month"]

    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    item = get_data(run_config.dataset_path, "item", run_config.suffix)
    customer = get_data(run_config.dataset_path, "customer", run_config.suffix)
    catalog_sales = get_data(
        run_config.dataset_path, "catalog_sales", run_config.suffix
    )
    web_sales = get_data(run_config.dataset_path, "web_sales", run_config.suffix)

    frequent_ss_items = (
        store_sales.join(date_dim, left_on="ss_sold_date_sk", right_on="d_date_sk")
        .join(item, left_on="ss_item_sk", right_on="i_item_sk")
        .filter(pl.col("d_year").is_in([year, year + 1, year + 2, year + 3]))
        .with_columns(pl.col("i_item_desc").str.slice(0, 30).alias("itemdesc"))
        .group_by(["itemdesc", "ss_item_sk", "d_date"])
        .agg(pl.len().alias("cnt"))
        .filter(pl.col("cnt") > 4)
        .select("ss_item_sk")
    )

    customer_sales = (
        # Note: filtering on is_not_null instead of joining to the customer table is
        # only valid because we know that the TPC-DS includes a foreign key here, so all
        # customers in store_sales _must_ be entries that exist somewhere in customer.
        store_sales.filter(pl.col("ss_customer_sk").is_not_null())
        .join(date_dim, left_on="ss_sold_date_sk", right_on="d_date_sk")
        .filter(pl.col("d_year").is_in([year, year + 1, year + 2, year + 3]))
        .group_by("ss_customer_sk")
        .agg((pl.col("ss_quantity") * pl.col("ss_sales_price")).sum().alias("csales"))
    )

    threshold = customer_sales.select(
        (pl.col("csales").max() * 0.95).alias("threshold")
    )

    best_customers = (
        store_sales.filter(pl.col("ss_customer_sk").is_not_null())
        .group_by("ss_customer_sk")
        .agg((pl.col("ss_quantity") * pl.col("ss_sales_price")).sum().alias("ssales"))
        .join(threshold, how="cross")
        .filter(pl.col("ssales") > pl.col("threshold"))
        .select("ss_customer_sk")
        .unique()
    )

    # Pre-filter date_dim to target year/month
    date_target = date_dim.filter(
        (pl.col("d_year") == year) & (pl.col("d_moy") == month)
    ).select("d_date_sk")

    catalog_part = (
        catalog_sales.join(
            customer.select(["c_customer_sk", "c_last_name", "c_first_name"]),
            left_on="cs_bill_customer_sk",
            right_on="c_customer_sk",
        )
        .join(date_target, left_on="cs_sold_date_sk", right_on="d_date_sk")
        .join(frequent_ss_items, left_on="cs_item_sk", right_on="ss_item_sk")
        .join(
            best_customers,
            left_on="cs_bill_customer_sk",
            right_on="ss_customer_sk",
            how="semi",
        )
        .group_by(["c_last_name", "c_first_name"])
        .agg((pl.col("cs_quantity") * pl.col("cs_list_price")).sum().alias("sales"))
    )

    web_part = (
        web_sales.join(
            customer.select(["c_customer_sk", "c_last_name", "c_first_name"]),
            left_on="ws_bill_customer_sk",
            right_on="c_customer_sk",
        )
        .join(date_target, left_on="ws_sold_date_sk", right_on="d_date_sk")
        .join(frequent_ss_items, left_on="ws_item_sk", right_on="ss_item_sk")
        .join(
            best_customers,
            left_on="ws_bill_customer_sk",
            right_on="ss_customer_sk",
            how="semi",
        )
        .group_by(["c_last_name", "c_first_name"])
        .agg((pl.col("ws_quantity") * pl.col("ws_list_price")).sum().alias("sales"))
    )

    sort_by = {"c_last_name": False, "c_first_name": False, "sales": False}
    limit = 100

    return QueryResult(
        frame=(
            pl.concat([catalog_part, web_part])
            .sort(sort_by.keys(), nulls_last=False)
            .limit(limit)
        ),
        sort_by=list(sort_by.items()),
        limit=limit,
        nulls_last=False,
    )


def polars_impl_naive(run_config: RunConfig) -> QueryResult:
    """Query 23 (naive)."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=23,
        qualification=run_config.qualification,
    )

    year = params["year"]
    month = params["month"]

    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    item = get_data(run_config.dataset_path, "item", run_config.suffix)
    customer = get_data(run_config.dataset_path, "customer", run_config.suffix)
    catalog_sales = get_data(
        run_config.dataset_path, "catalog_sales", run_config.suffix
    )
    web_sales = get_data(run_config.dataset_path, "web_sales", run_config.suffix)

    # SQL: CTE frequent_ss_items — store_sales items with count > 4 within year range
    frequent_ss_items = (
        # SQL: FROM store_sales, date_dim WHERE ss_sold_date_sk = d_date_sk AND d_year IN (...)
        store_sales.join(date_dim, left_on="ss_sold_date_sk", right_on="d_date_sk")
        # SQL: JOIN item ON ss_item_sk = i_item_sk (with SUBSTR(i_item_desc,1,30))
        .join(
            item.with_columns(pl.col("i_item_desc").str.slice(0, 30).alias("itemdesc")),
            left_on="ss_item_sk",
            right_on="i_item_sk",
        )
        # SQL: WHERE d_year IN ({year}, ..., {year}+3)
        .filter(pl.col("d_year").is_in([year, year + 1, year + 2, year + 3]))
        # SQL: GROUP BY itemdesc, i_item_sk, d_date HAVING count(*) > 4
        .group_by(["itemdesc", "ss_item_sk", "d_date"])
        .agg(pl.len().alias("cnt"))
        .filter(pl.col("cnt") > 4)
        .select(
            [
                pl.col("itemdesc"),
                pl.col("ss_item_sk").alias("item_sk"),
                pl.col("d_date").alias("solddate"),
                pl.col("cnt"),
            ]
        )
    )

    # SQL: CTE max_store_sales — max(csales) from customer store sales in year range
    max_store_sales = (
        # SQL: FROM store_sales, customer WHERE ss_customer_sk = c_customer_sk
        store_sales.join(
            customer, left_on="ss_customer_sk", right_on="c_customer_sk", suffix="_c"
        )
        # SQL: JOIN date_dim WHERE d_year IN (...) GROUP BY ss_customer_sk
        .join(date_dim, left_on="ss_sold_date_sk", right_on="d_date_sk")
        .filter(pl.col("d_year").is_in([year, year + 1, year + 2, year + 3]))
        .group_by("ss_customer_sk")
        .agg((pl.col("ss_quantity") * pl.col("ss_sales_price")).sum().alias("csales"))
        .select(pl.col("csales").max().alias("tpcds_cmax"))
    )

    # SQL: CTE best_ss_customer — customers with ssales > 95% of max
    best_ss_customer = (
        # SQL: FROM store_sales, customer, max_store_sales GROUP BY c_customer_sk HAVING ssales > 0.95 * max(tpcds_cmax)
        store_sales.join(
            customer, left_on="ss_customer_sk", right_on="c_customer_sk", suffix="_c"
        )
        .join(max_store_sales, how="cross")
        .group_by("ss_customer_sk")
        .agg(
            [
                (pl.col("ss_quantity") * pl.col("ss_sales_price"))
                .sum()
                .alias("ssales"),
                pl.col("tpcds_cmax").max().alias("tpcds_cmax"),
            ]
        )
        .filter(pl.col("ssales") > (pl.col("tpcds_cmax") * 0.95))
        .select(pl.col("ss_customer_sk").alias("c_customer_sk"))
    )

    # SQL: catalog_sales part — FROM catalog_sales, customer, date_dim, frequent_ss_items, best_ss_customer WHERE d_year={year} AND d_moy={month}
    catalog_part = (
        # SQL: JOIN customer ON cs_bill_customer_sk = c_customer_sk
        catalog_sales.join(
            customer, left_on="cs_bill_customer_sk", right_on="c_customer_sk"
        )
        # SQL: JOIN date_dim ON cs_sold_date_sk = d_date_sk
        .join(date_dim, left_on="cs_sold_date_sk", right_on="d_date_sk")
        # SQL: JOIN frequent_ss_items ON cs_item_sk = item_sk
        .join(frequent_ss_items, left_on="cs_item_sk", right_on="item_sk")
        # SQL: JOIN best_ss_customer ON cs_bill_customer_sk = c_customer_sk
        .join(best_ss_customer, left_on="cs_bill_customer_sk", right_on="c_customer_sk")
        # SQL: WHERE d_year = {year} AND d_moy = {month}
        .filter((pl.col("d_year") == year) & (pl.col("d_moy") == month))
        # SQL: GROUP BY c_last_name, c_first_name; Sum(cs_quantity*cs_list_price) AS sales
        .group_by(["c_last_name", "c_first_name"])
        .agg((pl.col("cs_quantity") * pl.col("cs_list_price")).sum().alias("sales"))
    )

    # SQL: web_sales part — FROM web_sales, customer, date_dim, frequent_ss_items, best_ss_customer WHERE d_year={year} AND d_moy={month}
    web_part = (
        # SQL: JOIN customer ON ws_bill_customer_sk = c_customer_sk
        web_sales.join(
            customer, left_on="ws_bill_customer_sk", right_on="c_customer_sk"
        )
        # SQL: JOIN date_dim ON ws_sold_date_sk = d_date_sk
        .join(date_dim, left_on="ws_sold_date_sk", right_on="d_date_sk")
        # SQL: JOIN frequent_ss_items ON ws_item_sk = item_sk
        .join(frequent_ss_items, left_on="ws_item_sk", right_on="item_sk")
        # SQL: JOIN best_ss_customer ON ws_bill_customer_sk = c_customer_sk
        .join(best_ss_customer, left_on="ws_bill_customer_sk", right_on="c_customer_sk")
        # SQL: WHERE d_year = {year} AND d_moy = {month}
        .filter((pl.col("d_year") == year) & (pl.col("d_moy") == month))
        # SQL: GROUP BY c_last_name, c_first_name; Sum(ws_quantity*ws_list_price) AS sales
        .group_by(["c_last_name", "c_first_name"])
        .agg((pl.col("ws_quantity") * pl.col("ws_list_price")).sum().alias("sales"))
    )

    sort_by = {"c_last_name": False, "c_first_name": False, "sales": False}
    limit = 100
    return QueryResult(
        frame=(
            # SQL: UNION ALL (catalog_part, web_part)
            pl.concat([catalog_part, web_part])
            # SQL: ORDER BY c_last_name, c_first_name, sales
            .sort(list(sort_by.keys()), nulls_last=False)
            # SQL: LIMIT 100
            .limit(limit)
        ),
        sort_by=list(sort_by.items()),
        limit=limit,
        nulls_last=False,
    )
