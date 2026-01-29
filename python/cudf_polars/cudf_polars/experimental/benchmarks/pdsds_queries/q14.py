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
       WITH cross_items AS
       (SELECT i_item_sk ss_item_sk
       FROM item,
       (SELECT iss.i_brand_id brand_id,
              iss.i_class_id class_id,
              iss.i_category_id category_id
       FROM store_sales,
              item iss,
              date_dim d1
       WHERE ss_item_sk = iss.i_item_sk
              AND ss_sold_date_sk = d1.d_date_sk
              AND d1.d_year BETWEEN 1999 AND 1999 + 2 INTERSECT
              SELECT ics.i_brand_id,
                     ics.i_class_id,
                     ics.i_category_id
              FROM catalog_sales,
              item ics,
              date_dim d2 WHERE cs_item_sk = ics.i_item_sk
              AND cs_sold_date_sk = d2.d_date_sk
              AND d2.d_year BETWEEN 1999 AND 1999 + 2 INTERSECT
              SELECT iws.i_brand_id,
                     iws.i_class_id,
                     iws.i_category_id
              FROM web_sales,
              item iws,
              date_dim d3 WHERE ws_item_sk = iws.i_item_sk
              AND ws_sold_date_sk = d3.d_date_sk
              AND d3.d_year BETWEEN 1999 AND 1999 + 2) sq1
       WHERE i_brand_id = brand_id
       AND i_class_id = class_id
       AND i_category_id = category_id ),
       avg_sales AS
       (SELECT avg(quantity*list_price) average_sales
       FROM
       (SELECT ss_quantity quantity,
              ss_list_price list_price
       FROM store_sales,
              date_dim
       WHERE ss_sold_date_sk = d_date_sk
              AND d_year BETWEEN 1999 AND 1999 + 2
       UNION ALL SELECT cs_quantity quantity,
                            cs_list_price list_price
       FROM catalog_sales,
              date_dim
       WHERE cs_sold_date_sk = d_date_sk
              AND d_year BETWEEN 1999 AND 1999 + 2
       UNION ALL SELECT ws_quantity quantity,
                            ws_list_price list_price
       FROM web_sales,
              date_dim
       WHERE ws_sold_date_sk = d_date_sk
              AND d_year BETWEEN 1999 AND 1999 + 2) sq2)
       SELECT channel,
              i_brand_id,
              i_class_id,
              i_category_id,
              sum(sales) AS sum_sales,
              sum(number_sales) AS sum_number_sales
       FROM
       (SELECT 'store' channel,
                     i_brand_id,
                     i_class_id,
                     i_category_id,
                     sum(ss_quantity*ss_list_price) sales,
                     count(*) number_sales
       FROM store_sales,
              item,
              date_dim
       WHERE ss_item_sk IN
              (SELECT ss_item_sk
              FROM cross_items)
       AND ss_item_sk = i_item_sk
       AND ss_sold_date_sk = d_date_sk
       AND d_year = 1999+2
       AND d_moy = 11
       GROUP BY i_brand_id,
              i_class_id,
              i_category_id
       HAVING sum(ss_quantity*ss_list_price) >
       (SELECT average_sales
       FROM avg_sales)
       UNION ALL SELECT 'catalog' channel,
                                   i_brand_id,
                                   i_class_id,
                                   i_category_id,
                                   sum(cs_quantity*cs_list_price) sales,
                                   count(*) number_sales
       FROM catalog_sales,
              item,
              date_dim
       WHERE cs_item_sk IN
              (SELECT ss_item_sk
              FROM cross_items)
       AND cs_item_sk = i_item_sk
       AND cs_sold_date_sk = d_date_sk
       AND d_year = 1999+2
       AND d_moy = 11
       GROUP BY i_brand_id,
              i_class_id,
              i_category_id
       HAVING sum(cs_quantity*cs_list_price) >
       (SELECT average_sales
       FROM avg_sales)
       UNION ALL SELECT 'web' channel,
                            i_brand_id,
                            i_class_id,
                            i_category_id,
                            sum(ws_quantity*ws_list_price) sales,
                            count(*) number_sales
       FROM web_sales,
              item,
              date_dim
       WHERE ws_item_sk IN
              (SELECT ss_item_sk
              FROM cross_items)
       AND ws_item_sk = i_item_sk
       AND ws_sold_date_sk = d_date_sk
       AND d_year = 1999+2
       AND d_moy = 11
       GROUP BY i_brand_id,
              i_class_id,
              i_category_id
       HAVING sum(ws_quantity*ws_list_price) >
       (SELECT average_sales
       FROM avg_sales)) y
       GROUP BY ROLLUP (channel,
                     i_brand_id,
                     i_class_id,
                     i_category_id)
       ORDER BY channel NULLS FIRST,
              i_brand_id NULLS FIRST,
              i_class_id NULLS FIRST,
              i_category_id NULLS FIRST
       LIMIT 100;
    """


def channel_items(  # noqa: D103
    sales: pl.LazyFrame,
    item: pl.LazyFrame,
    date_dim: pl.LazyFrame,
    *,
    item_key: str,
    date_key: str,
) -> pl.LazyFrame:
    return (
        sales.join(item, left_on=item_key, right_on="i_item_sk")
        .join(date_dim, left_on=date_key, right_on="d_date_sk")
        .filter(pl.col("d_year").is_between(1999, 2001))
        .select(["i_brand_id", "i_class_id", "i_category_id"])
        .unique()
    )


def build_cross_items(  # noqa: D103
    store_sales: pl.LazyFrame,
    catalog_sales: pl.LazyFrame,
    web_sales: pl.LazyFrame,
    item: pl.LazyFrame,
    date_dim: pl.LazyFrame,
) -> pl.LazyFrame:
    store_items = channel_items(
        store_sales, item, date_dim, item_key="ss_item_sk", date_key="ss_sold_date_sk"
    )
    catalog_items = channel_items(
        catalog_sales, item, date_dim, item_key="cs_item_sk", date_key="cs_sold_date_sk"
    )
    web_items = channel_items(
        web_sales, item, date_dim, item_key="ws_item_sk", date_key="ws_sold_date_sk"
    )
    common = store_items.join(
        catalog_items, on=["i_brand_id", "i_class_id", "i_category_id"]
    ).join(web_items, on=["i_brand_id", "i_class_id", "i_category_id"])
    return item.join(common, on=["i_brand_id", "i_class_id", "i_category_id"]).select(
        pl.col("i_item_sk").alias("ss_item_sk")
    )


def avg_pairs(  # noqa: D103
    sales: pl.LazyFrame,
    date_dim: pl.LazyFrame,
    *,
    qty_col: str,
    price_col: str,
    date_key: str,
) -> pl.LazyFrame:
    return (
        sales.join(date_dim, left_on=date_key, right_on="d_date_sk")
        .filter(pl.col("d_year").is_between(1999, 2001))
        .select(
            [pl.col(qty_col).alias("quantity"), pl.col(price_col).alias("list_price")]
        )
    )


def build_average_sales(  # noqa: D103
    store_sales: pl.LazyFrame,
    catalog_sales: pl.LazyFrame,
    web_sales: pl.LazyFrame,
    date_dim: pl.LazyFrame,
) -> pl.LazyFrame:
    store_avg = avg_pairs(
        store_sales,
        date_dim,
        qty_col="ss_quantity",
        price_col="ss_list_price",
        date_key="ss_sold_date_sk",
    )
    catalog_avg = avg_pairs(
        catalog_sales,
        date_dim,
        qty_col="cs_quantity",
        price_col="cs_list_price",
        date_key="cs_sold_date_sk",
    )
    web_avg = avg_pairs(
        web_sales,
        date_dim,
        qty_col="ws_quantity",
        price_col="ws_list_price",
        date_key="ws_sold_date_sk",
    )
    return (
        pl.concat([store_avg, catalog_avg, web_avg])
        .with_columns((pl.col("quantity") * pl.col("list_price")).alias("sales_amount"))
        .select(pl.col("sales_amount").mean().alias("average_sales"))
    )


def build_channel_result(  # noqa: D103
    sales: pl.LazyFrame,
    item: pl.LazyFrame,
    date_dim: pl.LazyFrame,
    cross_items: pl.LazyFrame,
    *,
    item_key: str,
    date_key: str,
    qty_col: str,
    price_col: str,
    channel_label: str,
    year: int,
    moy: int,
    average_sales: pl.LazyFrame,
) -> pl.LazyFrame:
    return (
        sales.join(cross_items, left_on=item_key, right_on="ss_item_sk")
        .join(item, left_on=item_key, right_on="i_item_sk")
        .join(date_dim, left_on=date_key, right_on="d_date_sk")
        .filter((pl.col("d_year") == year) & (pl.col("d_moy") == moy))
        .group_by(["i_brand_id", "i_class_id", "i_category_id"])
        .agg(
            [
                (pl.col(qty_col) * pl.col(price_col)).sum().alias("sales"),
                pl.len().alias("number_sales"),
            ]
        )
        .join(average_sales, how="cross")
        .filter(pl.col("sales") > pl.col("average_sales"))
        .with_columns(pl.lit(channel_label).alias("channel"))
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


def rollup_level(y: pl.LazyFrame, group_cols: list[str]) -> pl.LazyFrame:  # noqa: D103
    if group_cols:
        lf = y.group_by(group_cols).agg(
            [
                pl.col("sales").sum().alias("sum_sales"),
                pl.col("number_sales").sum().alias("sum_number_sales"),
            ]
        )
    else:
        lf = y.select(
            [
                pl.col("sales").sum().alias("sum_sales"),
                pl.col("number_sales").sum().alias("sum_number_sales"),
            ]
        )

    cols = {
        "channel": pl.Utf8,
        "i_brand_id": pl.Int32,
        "i_class_id": pl.Int32,
        "i_category_id": pl.Int32,
    }
    missing = [c for c in cols if c not in group_cols]
    if missing:
        lf = lf.with_columns([pl.lit(None, dtype=cols[c]).alias(c) for c in missing])

    return lf.select(
        [
            "channel",
            "i_brand_id",
            "i_class_id",
            "i_category_id",
            "sum_sales",
            "sum_number_sales",
        ]
    )


def polars_impl(run_config: RunConfig) -> pl.LazyFrame:
    """Query 14."""
    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    catalog_sales = get_data(
        run_config.dataset_path, "catalog_sales", run_config.suffix
    )
    web_sales = get_data(run_config.dataset_path, "web_sales", run_config.suffix)
    item = get_data(run_config.dataset_path, "item", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)

    cross_items = build_cross_items(
        store_sales, catalog_sales, web_sales, item, date_dim
    )
    average_sales = build_average_sales(store_sales, catalog_sales, web_sales, date_dim)

    y_store = build_channel_result(
        store_sales,
        item,
        date_dim,
        cross_items,
        item_key="ss_item_sk",
        date_key="ss_sold_date_sk",
        qty_col="ss_quantity",
        price_col="ss_list_price",
        channel_label="store",
        year=2001,
        moy=11,
        average_sales=average_sales,
    )
    y_catalog = build_channel_result(
        catalog_sales,
        item,
        date_dim,
        cross_items,
        item_key="cs_item_sk",
        date_key="cs_sold_date_sk",
        qty_col="cs_quantity",
        price_col="cs_list_price",
        channel_label="catalog",
        year=2001,
        moy=11,
        average_sales=average_sales,
    )
    y_web = build_channel_result(
        web_sales,
        item,
        date_dim,
        cross_items,
        item_key="ws_item_sk",
        date_key="ws_sold_date_sk",
        qty_col="ws_quantity",
        price_col="ws_list_price",
        channel_label="web",
        year=2001,
        moy=11,
        average_sales=average_sales,
    )

    y = pl.concat([y_store, y_catalog, y_web])

    level1 = rollup_level(y, ["channel", "i_brand_id", "i_class_id", "i_category_id"])
    level2 = rollup_level(y, ["channel", "i_brand_id", "i_class_id"])
    level3 = rollup_level(y, ["channel", "i_brand_id"])
    level4 = rollup_level(y, ["channel"])
    level5 = rollup_level(y, [])

    return (
        pl.concat([level1, level2, level3, level4, level5])
        .sort(
            ["channel", "i_brand_id", "i_class_id", "i_category_id"], nulls_last=False
        )
        .limit(100)
    )
