# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 84."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.utils import get_data

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 84."""
    return """
    SELECT c_customer_id   AS customer_id,
                   c_last_name
                   || ', '
                   || c_first_name AS customername
    FROM   customer,
           customer_address,
           customer_demographics,
           household_demographics,
           income_band,
           store_returns
    WHERE  ca_city = 'Green Acres'
           AND c_current_addr_sk = ca_address_sk
           AND ib_lower_bound >= 54986
           AND ib_upper_bound <= 54986 + 50000
           AND ib_income_band_sk = hd_income_band_sk
           AND cd_demo_sk = c_current_cdemo_sk
           AND hd_demo_sk = c_current_hdemo_sk
           AND sr_cdemo_sk = cd_demo_sk
    ORDER  BY c_customer_id
    LIMIT 100;
    """


def polars_impl(run_config: RunConfig) -> pl.LazyFrame:
    """Query 84."""
    customer = get_data(run_config.dataset_path, "customer", run_config.suffix)
    customer_address = get_data(
        run_config.dataset_path, "customer_address", run_config.suffix
    )
    customer_demographics = get_data(
        run_config.dataset_path, "customer_demographics", run_config.suffix
    )
    household_demographics = get_data(
        run_config.dataset_path, "household_demographics", run_config.suffix
    )
    income_band = get_data(run_config.dataset_path, "income_band", run_config.suffix)
    store_returns = get_data(
        run_config.dataset_path, "store_returns", run_config.suffix
    )
    return (
        customer.join(
            customer_address.filter(pl.col("ca_city") == "Green Acres"),
            left_on="c_current_addr_sk",
            right_on="ca_address_sk",
            how="inner",
        )
        .join(
            customer_demographics,
            left_on="c_current_cdemo_sk",
            right_on="cd_demo_sk",
            how="inner",
        )
        .join(
            household_demographics,
            left_on="c_current_hdemo_sk",
            right_on="hd_demo_sk",
            how="inner",
        )
        .join(
            income_band.filter(
                (pl.col("ib_lower_bound") >= 54986)
                & (pl.col("ib_upper_bound") <= 54986 + 50000)
            ),
            left_on="hd_income_band_sk",
            right_on="ib_income_band_sk",
            how="inner",
        )
        .join(
            store_returns,
            left_on="c_current_cdemo_sk",
            right_on="sr_cdemo_sk",
            how="inner",
        )
        .select(
            [
                pl.col("c_customer_id").alias("customer_id"),
                (pl.col("c_last_name") + pl.lit(", ") + pl.col("c_first_name")).alias(
                    "customername"
                ),
            ]
        )
        .sort("customer_id", nulls_last=True)
        .limit(100)
    )
