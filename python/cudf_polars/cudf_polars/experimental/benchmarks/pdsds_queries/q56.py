# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 56."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.pdsds_parameters import load_parameters
from cudf_polars.experimental.benchmarks.polars_naive_helpers import channel_agg
from cudf_polars.experimental.benchmarks.utils import QueryResult, get_data

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 56."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=56,
        qualification=run_config.qualification,
    )

    year = params["year"]
    month = params["month"]
    colors = params["colors"]
    gmt_offset = params["gmt_offset"]

    colors_str = ", ".join(f"'{c}'" for c in colors)

    return f"""
        WITH ss
            AS (SELECT i_item_id,
                        Sum(ss_ext_sales_price) total_sales
                FROM   store_sales,
                        date_dim,
                        customer_address,
                        item
                WHERE  i_item_id IN (SELECT i_item_id
                                    FROM   item
                                    WHERE  i_color IN ( {colors_str} )
                                    )
                        AND ss_item_sk = i_item_sk
                        AND ss_sold_date_sk = d_date_sk
                        AND d_year = {year}
                        AND d_moy = {month}
                        AND ss_addr_sk = ca_address_sk
                        AND ca_gmt_offset = {gmt_offset}
                GROUP  BY i_item_id),
            cs
            AS (SELECT i_item_id,
                        Sum(cs_ext_sales_price) total_sales
                FROM   catalog_sales,
                        date_dim,
                        customer_address,
                        item
                WHERE  i_item_id IN (SELECT i_item_id
                                    FROM   item
                                    WHERE  i_color IN ( {colors_str} )
                                    )
                        AND cs_item_sk = i_item_sk
                        AND cs_sold_date_sk = d_date_sk
                        AND d_year = {year}
                        AND d_moy = {month}
                        AND cs_bill_addr_sk = ca_address_sk
                        AND ca_gmt_offset = {gmt_offset}
                GROUP  BY i_item_id),
            ws
            AS (SELECT i_item_id,
                        Sum(ws_ext_sales_price) total_sales
                FROM   web_sales,
                        date_dim,
                        customer_address,
                        item
                WHERE  i_item_id IN (SELECT i_item_id
                                    FROM   item
                                    WHERE  i_color IN ( {colors_str} )
                                    )
                        AND ws_item_sk = i_item_sk
                        AND ws_sold_date_sk = d_date_sk
                        AND d_year = {year}
                        AND d_moy = {month}
                        AND ws_bill_addr_sk = ca_address_sk
                        AND ca_gmt_offset = {gmt_offset}
                GROUP  BY i_item_id)
        SELECT i_item_id,
                    Sum(total_sales) total_sales
        FROM   (SELECT *
                FROM   ss
                UNION ALL
                SELECT *
                FROM   cs
                UNION ALL
                SELECT *
                FROM   ws) tmp1
        GROUP  BY i_item_id
        ORDER  BY total_sales
        LIMIT 100;
    """


def polars_impl(run_config: RunConfig) -> QueryResult:
    """Query 56."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=56,
        qualification=run_config.qualification,
    )

    year = params["year"]
    month = params["month"]
    colors = params["colors"]
    gmt_offset = params["gmt_offset"]

    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    catalog_sales = get_data(
        run_config.dataset_path, "catalog_sales", run_config.suffix
    )
    web_sales = get_data(run_config.dataset_path, "web_sales", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    customer_address = get_data(
        run_config.dataset_path, "customer_address", run_config.suffix
    )
    item = get_data(run_config.dataset_path, "item", run_config.suffix)

    color_item_ids_lf = (
        item.filter(pl.col("i_color").is_in(colors)).select(["i_item_id"]).unique()
    )

    channels: list[tuple[pl.LazyFrame, str, str, str, str]] = [
        (
            store_sales,
            "ss_sold_date_sk",
            "ss_item_sk",
            "ss_addr_sk",
            "ss_ext_sales_price",
        ),
        (
            catalog_sales,
            "cs_sold_date_sk",
            "cs_item_sk",
            "cs_bill_addr_sk",
            "cs_ext_sales_price",
        ),
        (
            web_sales,
            "ws_sold_date_sk",
            "ws_item_sk",
            "ws_bill_addr_sk",
            "ws_ext_sales_price",
        ),
    ]

    per_channel = [
        (
            lf.join(item, left_on=item_sk_col, right_on="i_item_sk")
            .join(color_item_ids_lf, on="i_item_id")
            .join(date_dim, left_on=sold_date_col, right_on="d_date_sk")
            .join(customer_address, left_on=addr_sk_col, right_on="ca_address_sk")
            .filter(
                (pl.col("d_year") == year)
                & (pl.col("d_moy") == month)
                & (pl.col("ca_gmt_offset") == gmt_offset)
            )
            .group_by("i_item_id")
            .agg([pl.col(ext_col).sum().alias("total_sales")])
            .select(["i_item_id", "total_sales"])
        )
        for lf, sold_date_col, item_sk_col, addr_sk_col, ext_col in channels
    ]

    sort_by = {"total_sales": False}
    limit = 100
    return QueryResult(
        frame=(
            pl.concat(per_channel)
            .group_by("i_item_id")
            .agg([pl.col("total_sales").sum().alias("total_sales")])
            .select(["i_item_id", "total_sales"])
            .sort(sort_by.keys(), nulls_last=True)
            .limit(limit)
        ),
        sort_by=list(sort_by.items()),
        limit=limit,
    )


def polars_impl_naive(run_config: RunConfig) -> QueryResult:
    """Query 56 (naive)."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=56,
        qualification=run_config.qualification,
    )

    year = params["year"]
    month = params["month"]
    colors = params["colors"]
    gmt_offset = params["gmt_offset"]

    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    catalog_sales = get_data(
        run_config.dataset_path, "catalog_sales", run_config.suffix
    )
    web_sales = get_data(run_config.dataset_path, "web_sales", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    customer_address = get_data(
        run_config.dataset_path, "customer_address", run_config.suffix
    )
    item = get_data(run_config.dataset_path, "item", run_config.suffix)

    # SQL: WHERE i_color IN ({colors}) — subquery for qualifying item IDs
    color_item_ids_lf = (
        item.filter(pl.col("i_color").is_in(colors)).select(["i_item_id"]).unique()
    )

    date_filter = (pl.col("d_year") == year) & (pl.col("d_moy") == month)
    gmt_filter = pl.col("ca_gmt_offset") == gmt_offset

    # SQL: CTE ss — store_sales JOIN date_dim, item, color_item_ids, customer_address WHERE d_year={year} AND d_moy={month} AND ca_gmt_offset={gmt_offset}; GROUP BY i_item_id; Sum(ss_ext_sales_price)
    ss = (
        channel_agg(
            store_sales,
            date_dim,
            sales_date_key="ss_sold_date_sk",
            date_filter=date_filter,
            entity_table=item,
            entity_key_sales="ss_item_sk",
            entity_key_dim="i_item_sk",
            extra_joins=[
                (color_item_ids_lf, "i_item_id", "i_item_id"),
                (customer_address, "ss_addr_sk", "ca_address_sk"),
            ],
            extra_filters=[gmt_filter],
            agg_exprs=[pl.col("ss_ext_sales_price").sum().alias("total_sales")],
            group_by_cols=["i_item_id"],
        )
        .select(["i_item_id", "total_sales"])
    )

    # SQL: CTE cs — catalog_sales JOIN date_dim, item, color_item_ids, customer_address; GROUP BY i_item_id; Sum(cs_ext_sales_price)
    cs = (
        channel_agg(
            catalog_sales,
            date_dim,
            sales_date_key="cs_sold_date_sk",
            date_filter=date_filter,
            entity_table=item,
            entity_key_sales="cs_item_sk",
            entity_key_dim="i_item_sk",
            extra_joins=[
                (color_item_ids_lf, "i_item_id", "i_item_id"),
                (customer_address, "cs_bill_addr_sk", "ca_address_sk"),
            ],
            extra_filters=[gmt_filter],
            agg_exprs=[pl.col("cs_ext_sales_price").sum().alias("total_sales")],
            group_by_cols=["i_item_id"],
        )
        .select(["i_item_id", "total_sales"])
    )

    # SQL: CTE ws — web_sales JOIN date_dim, item, color_item_ids, customer_address; GROUP BY i_item_id; Sum(ws_ext_sales_price)
    ws = (
        channel_agg(
            web_sales,
            date_dim,
            sales_date_key="ws_sold_date_sk",
            date_filter=date_filter,
            entity_table=item,
            entity_key_sales="ws_item_sk",
            entity_key_dim="i_item_sk",
            extra_joins=[
                (color_item_ids_lf, "i_item_id", "i_item_id"),
                (customer_address, "ws_bill_addr_sk", "ca_address_sk"),
            ],
            extra_filters=[gmt_filter],
            agg_exprs=[pl.col("ws_ext_sales_price").sum().alias("total_sales")],
            group_by_cols=["i_item_id"],
        )
        .select(["i_item_id", "total_sales"])
    )

    sort_by = {"total_sales": False}
    limit = 100

    return QueryResult(
        frame=(
            # SQL: UNION ALL (ss, cs, ws)
            pl.concat([ss, cs, ws])
            # SQL: GROUP BY i_item_id; Sum(total_sales) total_sales
            .group_by("i_item_id")
            .agg([pl.col("total_sales").sum().alias("total_sales")])
            # SQL: SELECT i_item_id, total_sales
            .select(["i_item_id", "total_sales"])
            # SQL: ORDER BY total_sales
            .sort(sort_by.keys(), nulls_last=True)
            # SQL: LIMIT 100
            .limit(limit)
        ),
        sort_by=list(sort_by.items()),
        limit=limit,
    )
