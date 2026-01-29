# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 83."""

from __future__ import annotations

from datetime import date as _date
from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.utils import get_data

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 83."""
    return """
    WITH sr_items
         AS (SELECT i_item_id               item_id,
                    Sum(sr_return_quantity) sr_item_qty
             FROM   store_returns,
                    item,
                    date_dim
             WHERE  sr_item_sk = i_item_sk
                    AND d_date IN (SELECT d_date
                                   FROM   date_dim
                                   WHERE  d_week_seq IN (SELECT d_week_seq
                                                         FROM   date_dim
                                                         WHERE
                                          d_date IN ( '1999-06-30',
                                                      '1999-08-28',
                                                      '1999-11-18'
                                                    )))
                    AND sr_returned_date_sk = d_date_sk
             GROUP  BY i_item_id),
         cr_items
         AS (SELECT i_item_id               item_id,
                    Sum(cr_return_quantity) cr_item_qty
             FROM   catalog_returns,
                    item,
                    date_dim
             WHERE  cr_item_sk = i_item_sk
                    AND d_date IN (SELECT d_date
                                   FROM   date_dim
                                   WHERE  d_week_seq IN (SELECT d_week_seq
                                                         FROM   date_dim
                                                         WHERE
                                          d_date IN ( '1999-06-30',
                                                      '1999-08-28',
                                                      '1999-11-18'
                                                    )))
                    AND cr_returned_date_sk = d_date_sk
             GROUP  BY i_item_id),
         wr_items
         AS (SELECT i_item_id               item_id,
                    Sum(wr_return_quantity) wr_item_qty
             FROM   web_returns,
                    item,
                    date_dim
             WHERE  wr_item_sk = i_item_sk
                    AND d_date IN (SELECT d_date
                                   FROM   date_dim
                                   WHERE  d_week_seq IN (SELECT d_week_seq
                                                         FROM   date_dim
                                                         WHERE
                                          d_date IN ( '1999-06-30',
                                                      '1999-08-28',
                                                      '1999-11-18'
                                                    )))
                    AND wr_returned_date_sk = d_date_sk
             GROUP  BY i_item_id)
    SELECT sr_items.item_id,
                   sr_item_qty,
                   sr_item_qty / ( sr_item_qty + cr_item_qty + wr_item_qty ) / 3.0 *
                   100 sr_dev,
                   cr_item_qty,
                   cr_item_qty / ( sr_item_qty + cr_item_qty + wr_item_qty ) / 3.0 *
                   100 cr_dev,
                   wr_item_qty,
                   wr_item_qty / ( sr_item_qty + cr_item_qty + wr_item_qty ) / 3.0 *
                   100 wr_dev,
                   ( sr_item_qty + cr_item_qty + wr_item_qty ) / 3.0
                   average
    FROM   sr_items,
           cr_items,
           wr_items
    WHERE  sr_items.item_id = cr_items.item_id
           AND sr_items.item_id = wr_items.item_id
    ORDER  BY sr_items.item_id,
              sr_item_qty
    LIMIT 100;
    """


def q83_segment(
    returns: pl.LazyFrame,
    item: pl.LazyFrame,
    dates: pl.LazyFrame,
    *,
    item_key: str,
    returned_date_key: str,
    qty_col: str,
    out_qty_name: str,
) -> pl.LazyFrame:
    """Aggregate a returns table to per-item quantities over selected dates."""
    return (
        returns.join(
            item.select(["i_item_sk", "i_item_id"]),
            left_on=item_key,
            right_on="i_item_sk",
        )
        .join(dates, left_on=returned_date_key, right_on="d_date_sk")
        .group_by("i_item_id")
        .agg(pl.col(qty_col).sum().alias(out_qty_name))
        .select(["i_item_id", out_qty_name])
    )


def polars_impl(run_config: RunConfig) -> pl.LazyFrame:
    """Query 83."""
    store_returns = get_data(
        run_config.dataset_path, "store_returns", run_config.suffix
    )
    catalog_returns = get_data(
        run_config.dataset_path, "catalog_returns", run_config.suffix
    )
    web_returns = get_data(run_config.dataset_path, "web_returns", run_config.suffix)
    item = get_data(run_config.dataset_path, "item", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)

    anchor_dates = [_date(1999, 6, 30), _date(1999, 8, 28), _date(1999, 11, 18)]
    weeks = (
        date_dim.filter(pl.col("d_date").is_in(anchor_dates))
        .select("d_week_seq")
        .unique()
    )
    dates = date_dim.join(weeks, on="d_week_seq").select("d_date_sk").unique()

    sr_items = q83_segment(
        store_returns,
        item,
        dates,
        item_key="sr_item_sk",
        returned_date_key="sr_returned_date_sk",
        qty_col="sr_return_quantity",
        out_qty_name="sr_item_qty",
    )
    cr_items = q83_segment(
        catalog_returns,
        item,
        dates,
        item_key="cr_item_sk",
        returned_date_key="cr_returned_date_sk",
        qty_col="cr_return_quantity",
        out_qty_name="cr_item_qty",
    )
    wr_items = q83_segment(
        web_returns,
        item,
        dates,
        item_key="wr_item_sk",
        returned_date_key="wr_returned_date_sk",
        qty_col="wr_return_quantity",
        out_qty_name="wr_item_qty",
    )

    return (
        sr_items.join(cr_items, on="i_item_id")
        .join(wr_items, on="i_item_id")
        .with_columns(
            (
                pl.col("sr_item_qty") + pl.col("cr_item_qty") + pl.col("wr_item_qty")
            ).alias("total_qty")
        )
        .with_columns(
            [
                (pl.col("total_qty") / 3.0).cast(pl.Float64).alias("average"),
                (pl.col("sr_item_qty") / pl.col("total_qty") / 3.0 * 100)
                .cast(pl.Float64)
                .alias("sr_dev"),
                (pl.col("cr_item_qty") / pl.col("total_qty") / 3.0 * 100)
                .cast(pl.Float64)
                .alias("cr_dev"),
                (pl.col("wr_item_qty") / pl.col("total_qty") / 3.0 * 100)
                .cast(pl.Float64)
                .alias("wr_dev"),
            ]
        )
        .select(
            [
                pl.col("i_item_id").alias("item_id"),
                "sr_item_qty",
                "sr_dev",
                "cr_item_qty",
                "cr_dev",
                "wr_item_qty",
                "wr_dev",
                "average",
            ]
        )
        .sort(["item_id", "sr_item_qty"])
        .limit(100)
    )
