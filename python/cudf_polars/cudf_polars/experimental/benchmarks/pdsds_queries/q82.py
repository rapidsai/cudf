# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 82."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.utils import get_data

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 82."""
    return """
    -- start query 82 in stream 0 using template query82.tpl
    SELECT
             i_item_id ,
             i_item_desc ,
             i_current_price
    FROM     item,
             inventory,
             date_dim,
             store_sales
    WHERE    i_current_price BETWEEN 63 AND      63+30
    AND      inv_item_sk = i_item_sk
    AND      d_date_sk=inv_date_sk
    AND      d_date BETWEEN Cast('1998-04-27' AS DATE) AND      (
                      Cast('1998-04-27' AS DATE) + INTERVAL '60' day)
    AND      i_manufact_id IN (57,293,427,320)
    AND      inv_quantity_on_hand BETWEEN 100 AND      500
    AND      ss_item_sk = i_item_sk
    GROUP BY i_item_id,
             i_item_desc,
             i_current_price
    ORDER BY i_item_id
    LIMIT 100;
    """


def polars_impl(run_config: RunConfig) -> pl.LazyFrame:
    """Query 82."""
    item = get_data(run_config.dataset_path, "item", run_config.suffix)
    inventory = get_data(run_config.dataset_path, "inventory", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    start_date = pl.date(1998, 4, 27)
    end_date = pl.date(1998, 6, 26)
    return (
        item.join(inventory, left_on="i_item_sk", right_on="inv_item_sk")
        .join(date_dim, left_on="inv_date_sk", right_on="d_date_sk")
        .join(store_sales, left_on="i_item_sk", right_on="ss_item_sk")
        .filter(
            (pl.col("i_current_price") >= 63)
            & (pl.col("i_current_price") <= 93)
            & (pl.col("d_date") >= start_date)
            & (pl.col("d_date") <= end_date)
            & pl.col("i_manufact_id").is_in([57, 293, 427, 320])
            & (pl.col("inv_quantity_on_hand") >= 100)
            & (pl.col("inv_quantity_on_hand") <= 500)
        )
        .group_by(["i_item_id", "i_item_desc", "i_current_price"])
        .agg([])
        .select(["i_item_id", "i_item_desc", "i_current_price"])
        .sort("i_item_id")
        .limit(100)
    )
