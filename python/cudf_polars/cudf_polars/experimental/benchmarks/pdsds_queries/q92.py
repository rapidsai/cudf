# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 92."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.pdsds_parameters import load_parameters
from cudf_polars.experimental.benchmarks.utils import get_data

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 92."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=92,
        qualification=run_config.qualification,
    )

    manufact_id = params["manufact_id"]
    date = params["date"]

    return f"""
    SELECT
             Sum(ws_ext_discount_amt) AS 'Excess Discount Amount'
    FROM     web_sales ,
             item ,
             date_dim
    WHERE    i_manufact_id = {manufact_id}
    AND      i_item_sk = ws_item_sk
    AND      d_date BETWEEN '{date}' AND      (
                      Cast('{date}' AS DATE) +  INTERVAL '90' day)
    AND      d_date_sk = ws_sold_date_sk
    AND      ws_ext_discount_amt >
             (
                    SELECT 1.3 * avg(ws_ext_discount_amt)
                    FROM   web_sales ,
                           date_dim
                    WHERE  ws_item_sk = i_item_sk
                    AND    d_date BETWEEN '{date}' AND    (
                                  cast('{date}' AS date) + INTERVAL '90' day)
                    AND    d_date_sk = ws_sold_date_sk )
    ORDER BY sum(ws_ext_discount_amt)
    LIMIT 100;
    """


def polars_impl(run_config: RunConfig) -> pl.LazyFrame:
    """Query 92."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=92,
        qualification=run_config.qualification,
    )

    manufact_id = params["manufact_id"]
    date = params["date"]

    web_sales = get_data(run_config.dataset_path, "web_sales", run_config.suffix)
    item = get_data(run_config.dataset_path, "item", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    start_date_py = datetime.strptime(date, "%Y-%m-%d")
    start_date = pl.lit(start_date_py, dtype=pl.Datetime("us"))
    end_date = start_date + pl.duration(days=90)
    avg_discounts = (
        web_sales.join(
            date_dim, left_on="ws_sold_date_sk", right_on="d_date_sk", how="inner"
        )
        .filter((pl.col("d_date") >= start_date) & (pl.col("d_date") <= end_date))
        .group_by("ws_item_sk")
        .agg([pl.col("ws_ext_discount_amt").mean().alias("avg_discount")])
        .with_columns([(pl.col("avg_discount") * 1.3).alias("threshold_discount")])
    )
    return (
        web_sales.join(item, left_on="ws_item_sk", right_on="i_item_sk", how="inner")
        .join(date_dim, left_on="ws_sold_date_sk", right_on="d_date_sk", how="inner")
        .join(avg_discounts, left_on="ws_item_sk", right_on="ws_item_sk", how="inner")
        .filter(
            (pl.col("i_manufact_id") == manufact_id)
            & (pl.col("d_date") >= start_date)
            & (pl.col("d_date") <= end_date)
            & (pl.col("ws_ext_discount_amt") > pl.col("threshold_discount"))
        )
        .select([pl.col("ws_ext_discount_amt").sum().alias("Excess Discount Amount")])
        .sort("Excess Discount Amount", nulls_last=True)
        .limit(100)
    )
