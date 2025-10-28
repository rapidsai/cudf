# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 37."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.utils import get_data

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 37."""
    return """
    SELECT
             i_item_id ,
             i_item_desc ,
             i_current_price
    FROM     item,
             inventory,
             date_dim,
             catalog_sales
    WHERE    i_current_price BETWEEN 20 AND      20 + 30
    AND      inv_item_sk = i_item_sk
    AND      d_date_sk=inv_date_sk
    AND      d_date BETWEEN Cast('1999-03-06' AS DATE) AND      (
                      Cast('1999-03-06' AS DATE) + INTERVAL '60' day)
    AND      i_manufact_id IN (843,815,850,840)
    AND      inv_quantity_on_hand BETWEEN 100 AND      500
    AND      cs_item_sk = i_item_sk
    GROUP BY i_item_id,
             i_item_desc,
             i_current_price
    ORDER BY i_item_id
    LIMIT 100;
    """


def polars_impl(run_config: RunConfig) -> pl.LazyFrame:
    """Query 37."""
    # Load tables
    item = get_data(run_config.dataset_path, "item", run_config.suffix)
    inventory = get_data(run_config.dataset_path, "inventory", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    catalog_sales = get_data(
        run_config.dataset_path, "catalog_sales", run_config.suffix
    )
    return (
        item.join(inventory, left_on="i_item_sk", right_on="inv_item_sk")
        .join(date_dim, left_on="inv_date_sk", right_on="d_date_sk")
        .join(catalog_sales, left_on="i_item_sk", right_on="cs_item_sk")
        .filter(
            (pl.col("i_current_price").is_between(20, 50))
            & (pl.col("i_manufact_id").is_in([843, 815, 850, 840]))
            & (pl.col("inv_quantity_on_hand").is_between(100, 500))
            & (pl.col("d_date").is_between(pl.date(1999, 3, 6), pl.date(1999, 5, 5)))
        )
        .group_by(["i_item_id", "i_item_desc", "i_current_price"])
        .agg([])
        .sort(["i_item_id"])
        .limit(100)
    )
