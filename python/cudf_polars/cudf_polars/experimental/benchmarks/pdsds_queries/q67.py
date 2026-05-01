# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 67."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.pdsds_parameters import load_parameters
from cudf_polars.experimental.benchmarks.polars_naive_helpers import rollup_level
from cudf_polars.experimental.benchmarks.utils import QueryResult, get_data

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 67."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=67,
        qualification=run_config.qualification,
    )

    dms = params["dms"]

    return f"""
        select *
        from (select i_category
                    ,i_class
                    ,i_brand
                    ,i_product_name
                    ,d_year
                    ,d_qoy
                    ,d_moy
                    ,s_store_id
                    ,sumsales
                    ,rank() over (partition by i_category order by sumsales desc) rk
            from (select i_category
                        ,i_class
                        ,i_brand
                        ,i_product_name
                        ,d_year
                        ,d_qoy
                        ,d_moy
                        ,s_store_id
                        ,sum(coalesce(ss_sales_price*ss_quantity,0)) sumsales
                    from store_sales
                        ,date_dim
                        ,store
                        ,item
            where  ss_sold_date_sk=d_date_sk
                and ss_item_sk=i_item_sk
                and ss_store_sk = s_store_sk
                and d_month_seq between {dms} and {dms}+11
            group by  rollup(i_category, i_class, i_brand, i_product_name, d_year, d_qoy, d_moy,s_store_id))dw1) dw2
        where rk <= 100
        order by i_category
                ,i_class
                ,i_brand
                ,i_product_name
                ,d_year
                ,d_qoy
                ,d_moy
                ,s_store_id
                ,sumsales
                ,rk
        limit 100;
    """


ALL_COLS: dict[str, pl.DataType] = {
    "i_category": pl.String(),
    "i_class": pl.String(),
    "i_brand": pl.String(),
    "i_product_name": pl.String(),
    "d_year": pl.Int64(),
    "d_qoy": pl.Int64(),
    "d_moy": pl.Int64(),
    "s_store_id": pl.String(),
}
AGG_EXPRS = [pl.col("sales_amount").sum().alias("sumsales")]
OUTPUT_ORDER = [
    "i_category",
    "i_class",
    "i_brand",
    "i_product_name",
    "d_year",
    "d_qoy",
    "d_moy",
    "s_store_id",
    "sumsales",
]


def polars_impl(run_config: RunConfig) -> QueryResult:
    """Query 67."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=67,
        qualification=run_config.qualification,
    )

    dms = params["dms"]

    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    store = get_data(run_config.dataset_path, "store", run_config.suffix)
    item = get_data(run_config.dataset_path, "item", run_config.suffix)

    # Pre-filter date_dim to the 12-month window before joining against store_sales [58].
    # d_month_seq not needed after filter; keep group-by output columns.
    filtered_dates = date_dim.filter(
        pl.col("d_month_seq").is_between(dms, dms + 11)
    ).select(["d_date_sk", "d_year", "d_qoy", "d_moy"])

    base_data = (
        store_sales.select(["ss_sold_date_sk", "ss_item_sk", "ss_store_sk", "ss_sales_price", "ss_quantity"])
        .join(filtered_dates, left_on="ss_sold_date_sk", right_on="d_date_sk")
        .join(store.select(["s_store_sk", "s_store_id"]), left_on="ss_store_sk", right_on="s_store_sk")
        .join(
            item.select(["i_item_sk", "i_category", "i_class", "i_brand", "i_product_name"]),
            left_on="ss_item_sk",
            right_on="i_item_sk",
        )
        .with_columns(
            (pl.col("ss_sales_price") * pl.col("ss_quantity"))
            .fill_null(0)
            .alias("sales_amount")
        )
    )

    level0 = rollup_level(
        base_data,
        [
            "i_category",
            "i_class",
            "i_brand",
            "i_product_name",
            "d_year",
            "d_qoy",
            "d_moy",
            "s_store_id",
        ],
        ALL_COLS,
        AGG_EXPRS,
        OUTPUT_ORDER,
    )
    level1 = rollup_level(
        base_data,
        [
            "i_category",
            "i_class",
            "i_brand",
            "i_product_name",
            "d_year",
            "d_qoy",
            "d_moy",
        ],
        ALL_COLS,
        AGG_EXPRS,
        OUTPUT_ORDER,
    )
    level2 = rollup_level(
        base_data,
        ["i_category", "i_class", "i_brand", "i_product_name", "d_year", "d_qoy"],
        ALL_COLS,
        AGG_EXPRS,
        OUTPUT_ORDER,
    )
    level3 = rollup_level(
        base_data,
        ["i_category", "i_class", "i_brand", "i_product_name", "d_year"],
        ALL_COLS,
        AGG_EXPRS,
        OUTPUT_ORDER,
    )
    level4 = rollup_level(
        base_data,
        ["i_category", "i_class", "i_brand", "i_product_name"],
        ALL_COLS,
        AGG_EXPRS,
        OUTPUT_ORDER,
    )
    level5 = rollup_level(
        base_data,
        ["i_category", "i_class", "i_brand"],
        ALL_COLS,
        AGG_EXPRS,
        OUTPUT_ORDER,
    )
    level6 = rollup_level(
        base_data,
        ["i_category", "i_class"],
        ALL_COLS,
        AGG_EXPRS,
        OUTPUT_ORDER,
    )
    level7 = rollup_level(
        base_data,
        ["i_category"],
        ALL_COLS,
        AGG_EXPRS,
        OUTPUT_ORDER,
    )
    level8 = rollup_level(
        base_data,
        [],
        ALL_COLS,
        AGG_EXPRS,
        OUTPUT_ORDER,
    )

    rollup_data = pl.concat(
        [level0, level1, level2, level3, level4, level5, level6, level7, level8]
    )

    ranked = rollup_data.with_columns(
        pl.col("sumsales")
        .rank(method="dense", descending=True)
        .over("i_category")
        .alias("rk")
    )

    return QueryResult(
        frame=(
            ranked.filter(pl.col("rk") <= 100)
            .sort(
                [
                    "i_category",
                    "i_class",
                    "i_brand",
                    "i_product_name",
                    "d_year",
                    "d_qoy",
                    "d_moy",
                    "s_store_id",
                    "sumsales",
                    "rk",
                ],
                nulls_last=True,
                descending=[False] * 10,
            )
            .limit(100)
        ),
        sort_by=[
            ("i_category", False),
            ("i_class", False),
            ("i_brand", False),
            ("i_product_name", False),
            ("d_year", False),
            ("d_qoy", False),
            ("d_moy", False),
            ("s_store_id", False),
            ("sumsales", False),
            ("rk", False),
        ],
        limit=100,
    )


def polars_impl_naive(run_config: RunConfig) -> QueryResult:
    """Query 67 (naive)."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=67,
        qualification=run_config.qualification,
    )

    dms = params["dms"]

    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    store = get_data(run_config.dataset_path, "store", run_config.suffix)
    item = get_data(run_config.dataset_path, "item", run_config.suffix)

    # SQL: base_data — FROM store_sales, date_dim, store, item WHERE d_month_seq BETWEEN {dms} AND {dms}+11; COALESCE(ss_sales_price*ss_quantity,0) AS sales_amount
    base_data = (
        # SQL: JOIN date_dim ON ss_sold_date_sk=d_date_sk; JOIN store ON ss_store_sk=s_store_sk; JOIN item ON ss_item_sk=i_item_sk
        store_sales.join(date_dim, left_on="ss_sold_date_sk", right_on="d_date_sk")
        .join(store, left_on="ss_store_sk", right_on="s_store_sk")
        .join(item, left_on="ss_item_sk", right_on="i_item_sk")
        .filter(pl.col("d_month_seq").is_between(dms, dms + 11))
        .with_columns(
            (pl.col("ss_sales_price") * pl.col("ss_quantity"))
            .fill_null(0)
            .alias("sales_amount")
        )
    )

    # SQL: ROLLUP(i_category, i_class, i_brand, i_product_name, d_year, d_qoy, d_moy, s_store_id) — 9 levels
    level0 = rollup_level(
        base_data,
        [
            "i_category",
            "i_class",
            "i_brand",
            "i_product_name",
            "d_year",
            "d_qoy",
            "d_moy",
            "s_store_id",
        ],
        ALL_COLS,
        AGG_EXPRS,
        OUTPUT_ORDER,
    )
    level1 = rollup_level(
        base_data,
        [
            "i_category",
            "i_class",
            "i_brand",
            "i_product_name",
            "d_year",
            "d_qoy",
            "d_moy",
        ],
        ALL_COLS,
        AGG_EXPRS,
        OUTPUT_ORDER,
    )
    level2 = rollup_level(
        base_data,
        ["i_category", "i_class", "i_brand", "i_product_name", "d_year", "d_qoy"],
        ALL_COLS,
        AGG_EXPRS,
        OUTPUT_ORDER,
    )
    level3 = rollup_level(
        base_data,
        ["i_category", "i_class", "i_brand", "i_product_name", "d_year"],
        ALL_COLS,
        AGG_EXPRS,
        OUTPUT_ORDER,
    )
    level4 = rollup_level(
        base_data,
        ["i_category", "i_class", "i_brand", "i_product_name"],
        ALL_COLS,
        AGG_EXPRS,
        OUTPUT_ORDER,
    )
    level5 = rollup_level(
        base_data,
        ["i_category", "i_class", "i_brand"],
        ALL_COLS,
        AGG_EXPRS,
        OUTPUT_ORDER,
    )
    level6 = rollup_level(
        base_data,
        ["i_category", "i_class"],
        ALL_COLS,
        AGG_EXPRS,
        OUTPUT_ORDER,
    )
    level7 = rollup_level(
        base_data,
        ["i_category"],
        ALL_COLS,
        AGG_EXPRS,
        OUTPUT_ORDER,
    )
    level8 = rollup_level(
        base_data,
        [],
        ALL_COLS,
        AGG_EXPRS,
        OUTPUT_ORDER,
    )

    # SQL: UNION ALL all rollup levels
    rollup_data = pl.concat(
        [level0, level1, level2, level3, level4, level5, level6, level7, level8]
    )

    # SQL: Rank() OVER (PARTITION BY i_category ORDER BY sumsales DESC) AS rk
    ranked = rollup_data.with_columns(
        pl.col("sumsales")
        .rank(method="min", descending=True)
        .over("i_category")
        .alias("rk")
    )

    return QueryResult(
        frame=(
            # SQL: WHERE rk <= 100
            ranked.filter(pl.col("rk") <= 100)
            # SQL: ORDER BY i_category, i_class, i_brand, i_product_name, d_year, d_qoy, d_moy, s_store_id, sumsales, rk
            .sort(
                [
                    "i_category",
                    "i_class",
                    "i_brand",
                    "i_product_name",
                    "d_year",
                    "d_qoy",
                    "d_moy",
                    "s_store_id",
                    "sumsales",
                    "rk",
                ],
                nulls_last=True,
                descending=[False] * 10,
            )
            # SQL: LIMIT 100
            .limit(100)
        ),
        sort_by=[
            ("i_category", False),
            ("i_class", False),
            ("i_brand", False),
            ("i_product_name", False),
            ("d_year", False),
            ("d_qoy", False),
            ("d_moy", False),
            ("s_store_id", False),
            ("sumsales", False),
            ("rk", False),
        ],
        limit=100,
    )
