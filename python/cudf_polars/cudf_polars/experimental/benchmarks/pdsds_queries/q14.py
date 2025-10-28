# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 14."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.utils import get_data

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 14."""
    return """
    WITH cross_items
         AS (SELECT i_item_sk ss_item_sk
             FROM   item,
                    (SELECT iss.i_brand_id    brand_id,
                            iss.i_class_id    class_id,
                            iss.i_category_id category_id
                     FROM   store_sales,
                            item iss,
                            date_dim d1
                     WHERE  ss_item_sk = iss.i_item_sk
                            AND ss_sold_date_sk = d1.d_date_sk
                            AND d1.d_year BETWEEN 1999 AND 1999 + 2
                     INTERSECT
                     SELECT ics.i_brand_id,
                            ics.i_class_id,
                            ics.i_category_id
                     FROM   catalog_sales,
                            item ics,
                            date_dim d2
                     WHERE  cs_item_sk = ics.i_item_sk
                            AND cs_sold_date_sk = d2.d_date_sk
                            AND d2.d_year BETWEEN 1999 AND 1999 + 2
                     INTERSECT
                     SELECT iws.i_brand_id,
                            iws.i_class_id,
                            iws.i_category_id
                     FROM   web_sales,
                            item iws,
                            date_dim d3
                     WHERE  ws_item_sk = iws.i_item_sk
                            AND ws_sold_date_sk = d3.d_date_sk
                            AND d3.d_year BETWEEN 1999 AND 1999 + 2)
             WHERE  i_brand_id = brand_id
                    AND i_class_id = class_id
                    AND i_category_id = category_id),
         avg_sales
         AS (SELECT Avg(quantity * list_price) average_sales
             FROM   (SELECT ss_quantity   quantity,
                            ss_list_price list_price
                     FROM   store_sales,
                            date_dim
                     WHERE  ss_sold_date_sk = d_date_sk
                            AND d_year BETWEEN 1999 AND 1999 + 2
                     UNION ALL
                     SELECT cs_quantity   quantity,
                            cs_list_price list_price
                     FROM   catalog_sales,
                            date_dim
                     WHERE  cs_sold_date_sk = d_date_sk
                            AND d_year BETWEEN 1999 AND 1999 + 2
                     UNION ALL
                     SELECT ws_quantity   quantity,
                            ws_list_price list_price
                     FROM   web_sales,
                            date_dim
                     WHERE  ws_sold_date_sk = d_date_sk
                            AND d_year BETWEEN 1999 AND 1999 + 2) x)
    SELECT channel,
                   i_brand_id,
                   i_class_id,
                   i_category_id,
                   Sum(sales),
                   Sum(number_sales)
    FROM  (SELECT 'store'                          channel,
                  i_brand_id,
                  i_class_id,
                  i_category_id,
                  Sum(ss_quantity * ss_list_price) sales,
                  Count(*)                         number_sales
           FROM   store_sales,
                  item,
                  date_dim
           WHERE  ss_item_sk IN (SELECT ss_item_sk
                                 FROM   cross_items)
                  AND ss_item_sk = i_item_sk
                  AND ss_sold_date_sk = d_date_sk
                  AND d_year = 1999 + 2
                  AND d_moy = 11
           GROUP  BY i_brand_id,
                     i_class_id,
                     i_category_id
           HAVING Sum(ss_quantity * ss_list_price) > (SELECT average_sales
                                                      FROM   avg_sales)
           UNION ALL
           SELECT 'catalog'                        channel,
                  i_brand_id,
                  i_class_id,
                  i_category_id,
                  Sum(cs_quantity * cs_list_price) sales,
                  Count(*)                         number_sales
           FROM   catalog_sales,
                  item,
                  date_dim
           WHERE  cs_item_sk IN (SELECT ss_item_sk
                                 FROM   cross_items)
                  AND cs_item_sk = i_item_sk
                  AND cs_sold_date_sk = d_date_sk
                  AND d_year = 1999 + 2
                  AND d_moy = 11
           GROUP  BY i_brand_id,
                     i_class_id,
                     i_category_id
           HAVING Sum(cs_quantity * cs_list_price) > (SELECT average_sales
                                                      FROM   avg_sales)
           UNION ALL
           SELECT 'web'                            channel,
                  i_brand_id,
                  i_class_id,
                  i_category_id,
                  Sum(ws_quantity * ws_list_price) sales,
                  Count(*)                         number_sales
           FROM   web_sales,
                  item,
                  date_dim
           WHERE  ws_item_sk IN (SELECT ss_item_sk
                                 FROM   cross_items)
                  AND ws_item_sk = i_item_sk
                  AND ws_sold_date_sk = d_date_sk
                  AND d_year = 1999 + 2
                  AND d_moy = 11
           GROUP  BY i_brand_id,
                     i_class_id,
                     i_category_id
           HAVING Sum(ws_quantity * ws_list_price) > (SELECT average_sales
                                                      FROM   avg_sales)) y
    GROUP  BY rollup ( channel, i_brand_id, i_class_id, i_category_id )
    ORDER  BY channel,
              i_brand_id,
              i_class_id,
              i_category_id
    LIMIT 100;

    WITH cross_items
         AS (SELECT i_item_sk ss_item_sk
             FROM   item,
                    (SELECT iss.i_brand_id    brand_id,
                            iss.i_class_id    class_id,
                            iss.i_category_id category_id
                     FROM   store_sales,
                            item iss,
                            date_dim d1
                     WHERE  ss_item_sk = iss.i_item_sk
                            AND ss_sold_date_sk = d1.d_date_sk
                            AND d1.d_year BETWEEN 1999 AND 1999 + 2
                     INTERSECT
                     SELECT ics.i_brand_id,
                            ics.i_class_id,
                            ics.i_category_id
                     FROM   catalog_sales,
                            item ics,
                            date_dim d2
                     WHERE  cs_item_sk = ics.i_item_sk
                            AND cs_sold_date_sk = d2.d_date_sk
                            AND d2.d_year BETWEEN 1999 AND 1999 + 2
                     INTERSECT
                     SELECT iws.i_brand_id,
                            iws.i_class_id,
                            iws.i_category_id
                     FROM   web_sales,
                            item iws,
                            date_dim d3
                     WHERE  ws_item_sk = iws.i_item_sk
                            AND ws_sold_date_sk = d3.d_date_sk
                            AND d3.d_year BETWEEN 1999 AND 1999 + 2) x
             WHERE  i_brand_id = brand_id
                    AND i_class_id = class_id
                    AND i_category_id = category_id),
         avg_sales
         AS (SELECT Avg(quantity * list_price) average_sales
             FROM   (SELECT ss_quantity   quantity,
                            ss_list_price list_price
                     FROM   store_sales,
                            date_dim
                     WHERE  ss_sold_date_sk = d_date_sk
                            AND d_year BETWEEN 1999 AND 1999 + 2
                     UNION ALL
                     SELECT cs_quantity   quantity,
                            cs_list_price list_price
                     FROM   catalog_sales,
                            date_dim
                     WHERE  cs_sold_date_sk = d_date_sk
                            AND d_year BETWEEN 1999 AND 1999 + 2
                     UNION ALL
                     SELECT ws_quantity   quantity,
                            ws_list_price list_price
                     FROM   web_sales,
                            date_dim
                     WHERE  ws_sold_date_sk = d_date_sk
                            AND d_year BETWEEN 1999 AND 1999 + 2) x)
    SELECT  *
    FROM   (SELECT 'store'                          channel,
                   i_brand_id,
                   i_class_id,
                   i_category_id,
                   Sum(ss_quantity * ss_list_price) sales,
                   Count(*)                         number_sales
            FROM   store_sales,
                   item,
                   date_dim
            WHERE  ss_item_sk IN (SELECT ss_item_sk
                                  FROM   cross_items)
                   AND ss_item_sk = i_item_sk
                   AND ss_sold_date_sk = d_date_sk
                   AND d_week_seq = (SELECT d_week_seq
                                     FROM   date_dim
                                     WHERE  d_year = 1999 + 1
                                            AND d_moy = 12
                                            AND d_dom = 25)
            GROUP  BY i_brand_id,
                      i_class_id,
                      i_category_id
            HAVING Sum(ss_quantity * ss_list_price) > (SELECT average_sales
                                                       FROM   avg_sales)) this_year,
           (SELECT 'store'                          channel,
                   i_brand_id,
                   i_class_id,
                   i_category_id,
                   Sum(ss_quantity * ss_list_price) sales,
                   Count(*)                         number_sales
            FROM   store_sales,
                   item,
                   date_dim
            WHERE  ss_item_sk IN (SELECT ss_item_sk
                                  FROM   cross_items)
                   AND ss_item_sk = i_item_sk
                   AND ss_sold_date_sk = d_date_sk
                   AND d_week_seq = (SELECT d_week_seq
                                     FROM   date_dim
                                     WHERE  d_year = 1999
                                            AND d_moy = 12
                                            AND d_dom = 25)
            GROUP  BY i_brand_id,
                      i_class_id,
                      i_category_id
            HAVING Sum(ss_quantity * ss_list_price) > (SELECT average_sales
                                                       FROM   avg_sales)) last_year
    WHERE  this_year.i_brand_id = last_year.i_brand_id
           AND this_year.i_class_id = last_year.i_class_id
           AND this_year.i_category_id = last_year.i_category_id
    ORDER  BY this_year.channel,
              this_year.i_brand_id,
              this_year.i_class_id,
              this_year.i_category_id
    LIMIT 100;
    """


def polars_impl(run_config: RunConfig) -> pl.LazyFrame:
    """Query 14."""
    # Load tables
    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    catalog_sales = get_data(
        run_config.dataset_path, "catalog_sales", run_config.suffix
    )
    web_sales = get_data(run_config.dataset_path, "web_sales", run_config.suffix)
    item = get_data(run_config.dataset_path, "item", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)

    # Step 1: Find items sold across all 3 channels (INTERSECT logic)
    # Store sales items (1999-2001)
    store_items = (
        store_sales.join(item, left_on="ss_item_sk", right_on="i_item_sk")
        .join(date_dim, left_on="ss_sold_date_sk", right_on="d_date_sk")
        .filter(pl.col("d_year").is_between(1999, 2001))
        .select(["i_brand_id", "i_class_id", "i_category_id"])
        .unique()
    )

    # Catalog sales items (1999-2001)
    catalog_items = (
        catalog_sales.join(item, left_on="cs_item_sk", right_on="i_item_sk")
        .join(date_dim, left_on="cs_sold_date_sk", right_on="d_date_sk")
        .filter(pl.col("d_year").is_between(1999, 2001))
        .select(["i_brand_id", "i_class_id", "i_category_id"])
        .unique()
    )

    # Web sales items (1999-2001)
    web_items = (
        web_sales.join(item, left_on="ws_item_sk", right_on="i_item_sk")
        .join(date_dim, left_on="ws_sold_date_sk", right_on="d_date_sk")
        .filter(pl.col("d_year").is_between(1999, 2001))
        .select(["i_brand_id", "i_class_id", "i_category_id"])
        .unique()
    )

    # Find intersection of all three channels
    common_items = store_items.join(
        catalog_items, on=["i_brand_id", "i_class_id", "i_category_id"]
    ).join(web_items, on=["i_brand_id", "i_class_id", "i_category_id"])

    # Get item_sk for cross_items (rename to ss_item_sk to match SQL alias)
    cross_items = item.join(
        common_items, on=["i_brand_id", "i_class_id", "i_category_id"]
    ).select(pl.col("i_item_sk").alias("ss_item_sk"))

    # Step 2: Calculate average sales across all channels (1999-2001)
    store_avg_data = (
        store_sales.join(date_dim, left_on="ss_sold_date_sk", right_on="d_date_sk")
        .filter(pl.col("d_year").is_between(1999, 2001))
        .select(
            [
                pl.col("ss_quantity").alias("quantity"),
                pl.col("ss_list_price").alias("list_price"),
            ]
        )
    )

    catalog_avg_data = (
        catalog_sales.join(date_dim, left_on="cs_sold_date_sk", right_on="d_date_sk")
        .filter(pl.col("d_year").is_between(1999, 2001))
        .select(
            [
                pl.col("cs_quantity").alias("quantity"),
                pl.col("cs_list_price").alias("list_price"),
            ]
        )
    )

    web_avg_data = (
        web_sales.join(date_dim, left_on="ws_sold_date_sk", right_on="d_date_sk")
        .filter(pl.col("d_year").is_between(1999, 2001))
        .select(
            [
                pl.col("ws_quantity").alias("quantity"),
                pl.col("ws_list_price").alias("list_price"),
            ]
        )
    )

    # Union all sales data and calculate average
    all_sales_data = pl.concat([store_avg_data, catalog_avg_data, web_avg_data])
    average_sales_table = all_sales_data.with_columns(
        (pl.col("quantity") * pl.col("list_price")).alias("sales_amount")
    ).select(pl.col("sales_amount").mean().alias("average_sales"))

    # d_week_seq for December 25, 2000
    this_year_week_table = (
        date_dim.filter(
            pl.col("d_week_seq").is_not_null()
            & (pl.col("d_year") == 2000)
            & (pl.col("d_moy") == 12)
            & (pl.col("d_dom") == 25)
        )
        .select("d_week_seq")
        .unique()
    )

    # d_week_seq for December 25, 1999
    last_year_week_table = (
        date_dim.filter(
            pl.col("d_week_seq").is_not_null()
            & (pl.col("d_year") == 1999)
            & (pl.col("d_moy") == 12)
            & (pl.col("d_dom") == 25)
        )
        .select("d_week_seq")
        .unique()
    )

    # This year store sales (2000)
    this_year_sales = (
        store_sales.join(cross_items, left_on="ss_item_sk", right_on="ss_item_sk")
        .join(item, left_on="ss_item_sk", right_on="i_item_sk")
        .join(date_dim, left_on="ss_sold_date_sk", right_on="d_date_sk")
        .join(this_year_week_table, on="d_week_seq", how="semi")
        .group_by(["i_brand_id", "i_class_id", "i_category_id"])
        .agg(
            [
                (pl.col("ss_quantity") * pl.col("ss_list_price")).sum().alias("sales"),
                pl.len().alias("number_sales"),
            ]
        )
        .join(average_sales_table, how="cross")
        .filter(pl.col("sales") > pl.col("average_sales"))
        .with_columns(pl.lit("store").alias("channel"))
        .select(
            [
                "channel",
                "i_brand_id",
                "i_class_id",
                "i_category_id",
                "sales",
                "number_sales",
            ]
        )
    )

    # Last year store sales (1999) - keep original column names for join
    last_year_sales = (
        store_sales.join(cross_items, left_on="ss_item_sk", right_on="ss_item_sk")
        .join(item, left_on="ss_item_sk", right_on="i_item_sk")
        .join(date_dim, left_on="ss_sold_date_sk", right_on="d_date_sk")
        .join(last_year_week_table, on="d_week_seq", how="semi")
        .group_by(["i_brand_id", "i_class_id", "i_category_id"])
        .agg(
            [
                (pl.col("ss_quantity") * pl.col("ss_list_price")).sum().alias("sales"),
                pl.len().alias("number_sales"),
            ]
        )
        .join(average_sales_table, how="cross")
        .filter(pl.col("sales") > pl.col("average_sales"))
        .with_columns(pl.lit("store").alias("channel"))
        .select(
            [
                "channel",
                "i_brand_id",
                "i_class_id",
                "i_category_id",
                "sales",
                "number_sales",
            ]
        )
    )

    # Join this_year and last_year
    return (
        this_year_sales.join(
            last_year_sales,
            on=["i_brand_id", "i_class_id", "i_category_id"],
            suffix="_1",
        )
        .with_columns(
            [
                # Add the missing _1 columns manually since join keys don't get suffixed
                pl.col("i_brand_id").alias("i_brand_id_1"),
                pl.col("i_class_id").alias("i_class_id_1"),
                pl.col("i_category_id").alias("i_category_id_1"),
            ]
        )
        .select(
            [
                "channel",
                "i_brand_id",
                "i_class_id",
                "i_category_id",
                "sales",
                # Cast -> Int64 to match DuckDB
                pl.col("number_sales").cast(pl.Int64),
                "channel_1",
                "i_brand_id_1",
                "i_class_id_1",
                "i_category_id_1",
                "sales_1",
                # Cast -> Int64 to match DuckDB
                pl.col("number_sales_1").cast(pl.Int64),
            ]
        )
        .sort(["channel", "i_brand_id", "i_class_id", "i_category_id"])
        .limit(100)
    )
