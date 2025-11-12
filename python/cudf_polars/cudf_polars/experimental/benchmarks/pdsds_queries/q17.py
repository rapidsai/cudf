# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 17."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.utils import get_data

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 17."""
    return """
    SELECT i_item_id,
                   i_item_desc,
                   s_state,
                   Count(ss_quantity)                                        AS
                   store_sales_quantitycount,
                   Avg(ss_quantity)                                          AS
                   store_sales_quantityave,
                   Stddev_samp(ss_quantity)                                  AS
                   store_sales_quantitystdev,
                   Stddev_samp(ss_quantity) / Avg(ss_quantity)               AS
                   store_sales_quantitycov,
                   Count(sr_return_quantity)                                 AS
                   store_returns_quantitycount,
                   Avg(sr_return_quantity)                                   AS
                   store_returns_quantityave,
                   Stddev_samp(sr_return_quantity)                           AS
                   store_returns_quantitystdev,
                   Stddev_samp(sr_return_quantity) / Avg(sr_return_quantity) AS
                   store_returns_quantitycov,
                   Count(cs_quantity)                                        AS
                   catalog_sales_quantitycount,
                   Avg(cs_quantity)                                          AS
                   catalog_sales_quantityave,
                   Stddev_samp(cs_quantity) / Avg(cs_quantity)               AS
                   catalog_sales_quantitystdev,
                   Stddev_samp(cs_quantity) / Avg(cs_quantity)               AS
                   catalog_sales_quantitycov
    FROM   store_sales,
           store_returns,
           catalog_sales,
           date_dim d1,
           date_dim d2,
           date_dim d3,
           store,
           item
    WHERE  d1.d_quarter_name = '1999Q1'
           AND d1.d_date_sk = ss_sold_date_sk
           AND i_item_sk = ss_item_sk
           AND s_store_sk = ss_store_sk
           AND ss_customer_sk = sr_customer_sk
           AND ss_item_sk = sr_item_sk
           AND ss_ticket_number = sr_ticket_number
           AND sr_returned_date_sk = d2.d_date_sk
           AND d2.d_quarter_name IN ( '1999Q1', '1999Q2', '1999Q3' )
           AND sr_customer_sk = cs_bill_customer_sk
           AND sr_item_sk = cs_item_sk
           AND cs_sold_date_sk = d3.d_date_sk
           AND d3.d_quarter_name IN ( '1999Q1', '1999Q2', '1999Q3' )
    GROUP  BY i_item_id,
              i_item_desc,
              s_state
    ORDER  BY i_item_id,
              i_item_desc,
              s_state
    LIMIT 100;
    """


def polars_impl(run_config: RunConfig) -> pl.LazyFrame:
    """Query 17."""
    # Load tables
    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    store_returns = get_data(
        run_config.dataset_path, "store_returns", run_config.suffix
    )
    catalog_sales = get_data(
        run_config.dataset_path, "catalog_sales", run_config.suffix
    )
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    store = get_data(run_config.dataset_path, "store", run_config.suffix)
    item = get_data(run_config.dataset_path, "item", run_config.suffix)

    # The SQL uses comma-separated joins which create a Cartesian product filtered by WHERE
    # We need to be more careful about preserving all valid combinations

    # First get the base combinations that satisfy the core relationship constraints
    store_sales_base = (
        store_sales.join(
            date_dim, left_on="ss_sold_date_sk", right_on="d_date_sk", suffix="_d1"
        )
        .join(item, left_on="ss_item_sk", right_on="i_item_sk")
        .join(store, left_on="ss_store_sk", right_on="s_store_sk")
        .filter(pl.col("d_quarter_name") == "1999Q1")
    )

    store_returns_base = store_returns.join(
        date_dim, left_on="sr_returned_date_sk", right_on="d_date_sk", suffix="_d2"
    ).filter(pl.col("d_quarter_name").is_in(["1999Q1", "1999Q2", "1999Q3"]))

    catalog_sales_base = catalog_sales.join(
        date_dim, left_on="cs_sold_date_sk", right_on="d_date_sk", suffix="_d3"
    ).filter(pl.col("d_quarter_name").is_in(["1999Q1", "1999Q2", "1999Q3"]))

    # Now create the full combination following the SQL logic
    return (
        store_sales_base.join(
            store_returns_base,
            left_on=["ss_customer_sk", "ss_item_sk", "ss_ticket_number"],
            right_on=["sr_customer_sk", "sr_item_sk", "sr_ticket_number"],
            how="inner",
            suffix="_sr",
        )  # This relationship must exist per SQL
        .join(
            catalog_sales_base,
            left_on=["ss_customer_sk", "ss_item_sk"],
            right_on=["cs_bill_customer_sk", "cs_item_sk"],
            how="inner",
            suffix="_cs",
        )  # This relationship must exist per SQL
        .group_by(["i_item_id", "i_item_desc", "s_state"])
        .agg(
            [
                # Cast -> Int64 to match DuckDB
                pl.col("ss_quantity")
                .count()
                .cast(pl.Int64)
                .alias("store_sales_quantitycount"),
                pl.col("ss_quantity").mean().alias("store_sales_quantityave"),
                pl.col("ss_quantity").std().alias("store_sales_quantitystdev"),
                (pl.col("ss_quantity").std() / pl.col("ss_quantity").mean()).alias(
                    "store_sales_quantitycov"
                ),
                # Cast -> Int64 to match DuckDB
                pl.col("sr_return_quantity")
                .count()
                .cast(pl.Int64)
                .alias("store_returns_quantitycount"),
                pl.col("sr_return_quantity").mean().alias("store_returns_quantityave"),
                pl.col("sr_return_quantity").std().alias("store_returns_quantitystdev"),
                (
                    pl.col("sr_return_quantity").std()
                    / pl.col("sr_return_quantity").mean()
                ).alias("store_returns_quantitycov"),
                # Cast -> Int64 to match DuckDB
                pl.col("cs_quantity")
                .count()
                .cast(pl.Int64)
                .alias("catalog_sales_quantitycount"),
                pl.col("cs_quantity").mean().alias("catalog_sales_quantityave"),
                pl.col("cs_quantity").std().alias("catalog_sales_quantitystdev"),
                (pl.col("cs_quantity").std() / pl.col("cs_quantity").mean()).alias(
                    "catalog_sales_quantitycov"
                ),
            ]
        )
        .sort(["i_item_id", "i_item_desc", "s_state"], nulls_last=True)
        .limit(100)
    )
