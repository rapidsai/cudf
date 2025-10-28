# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 67."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.utils import get_data

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 67."""
    return """
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
                and d_month_seq between 1181 and 1181+11
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


def _build_rollup_levels(
    base_data: pl.LazyFrame,
    full_cols: list[str],
    rollup_specs: list[tuple[list[str], dict[str, pl.DataType]]],
) -> pl.LazyFrame:
    levels: list[pl.LazyFrame] = []

    for grp_cols, null_cols in rollup_specs:
        lf = (
            base_data.group_by(grp_cols).agg(pl.col("sales_amount").sum().alias("sumsales"))
            if grp_cols
            else base_data.select(pl.col("sales_amount").sum().alias("sumsales"))
        )

        if null_cols:
            lf = lf.with_columns([pl.lit(None, dtype=dt).alias(col) for col, dt in null_cols.items()])


        lf = lf.select([*(c for c in full_cols if c in grp_cols or c in null_cols), "sumsales"])

        missing = [c for c in full_cols if c not in grp_cols and c not in null_cols]
        if missing:
            lf = lf.with_columns([pl.lit(None).alias(c) for c in missing]).select(full_cols + ["sumsales"])

        levels.append(lf)

    return pl.concat(levels)


rollup_specs: list[tuple[list[str], dict[str, pl.DataType]]] = [
    (["i_category", "i_class", "i_brand", "i_product_name", "d_year", "d_qoy", "d_moy", "s_store_id"], {}),
    (["i_category", "i_class", "i_brand", "i_product_name", "d_year", "d_qoy", "d_moy"], {"s_store_id": pl.String}),
    (["i_category", "i_class", "i_brand", "i_product_name", "d_year", "d_qoy"], {"d_moy": pl.Int32, "s_store_id": pl.String}),
    (["i_category", "i_class", "i_brand", "i_product_name", "d_year"], {"d_qoy": pl.Int32, "d_moy": pl.Int32, "s_store_id": pl.String}),
    (["i_category", "i_class", "i_brand", "i_product_name"], {"d_year": pl.Int32, "d_qoy": pl.Int32, "d_moy": pl.Int32, "s_store_id": pl.String}),
    (["i_category", "i_class", "i_brand"], {"i_product_name": pl.String, "d_year": pl.Int32, "d_qoy": pl.Int32, "d_moy": pl.Int32, "s_store_id": pl.String}),
    (["i_category", "i_class"], {"i_brand": pl.String, "i_product_name": pl.String, "d_year": pl.Int32, "d_qoy": pl.Int32, "d_moy": pl.Int32, "s_store_id": pl.String}),
    (["i_category"], {"i_class": pl.String, "i_brand": pl.String, "i_product_name": pl.String, "d_year": pl.Int32, "d_qoy": pl.Int32, "d_moy": pl.Int32, "s_store_id": pl.String}),
    ([], {"i_category": pl.String, "i_class": pl.String, "i_brand": pl.String, "i_product_name": pl.String, "d_year": pl.Int32, "d_qoy": pl.Int32, "d_moy": pl.Int32, "s_store_id": pl.String}),
]


def polars_impl(run_config: RunConfig) -> pl.LazyFrame:
    """Query 67."""
    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    store = get_data(run_config.dataset_path, "store", run_config.suffix)
    item = get_data(run_config.dataset_path, "item", run_config.suffix)

    base_data = (
        store_sales
        .join(date_dim, left_on="ss_sold_date_sk", right_on="d_date_sk")
        .join(store, left_on="ss_store_sk", right_on="s_store_sk")
        .join(item, left_on="ss_item_sk", right_on="i_item_sk")
        .filter(pl.col("d_month_seq").is_between(1181, 1181 + 11))
        .with_columns((pl.col("ss_sales_price") * pl.col("ss_quantity")).fill_null(0).alias("sales_amount"))
    )

    full_cols = ["i_category", "i_class", "i_brand", "i_product_name", "d_year", "d_qoy", "d_moy", "s_store_id"]

    rollup_data = _build_rollup_levels(base_data, full_cols, rollup_specs)

    ranked = rollup_data.with_columns(
        pl.col("sumsales").rank(method="dense", descending=True).over("i_category").cast(pl.Int64).alias("rk")
    )

    return (
        ranked
        .filter(pl.col("rk") <= 100)
        .sort(
            ["i_category", "i_class", "i_brand", "i_product_name", "d_year", "d_qoy", "d_moy", "s_store_id", "sumsales", "rk"],
            nulls_last=True,
            descending=[False] * 10,
        )
        .limit(100)
    )
