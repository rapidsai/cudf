# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 91."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.utils import get_data

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 91."""
    return """
    SELECT cc_call_center_id Call_Center,
           cc_name           Call_Center_Name,
           cc_manager        Manager,
           Sum(cr_net_loss)  Returns_Loss
    FROM   call_center,
           catalog_returns,
           date_dim,
           customer,
           customer_address,
           customer_demographics,
           household_demographics
    WHERE  cr_call_center_sk = cc_call_center_sk
           AND cr_returned_date_sk = d_date_sk
           AND cr_returning_customer_sk = c_customer_sk
           AND cd_demo_sk = c_current_cdemo_sk
           AND hd_demo_sk = c_current_hdemo_sk
           AND ca_address_sk = c_current_addr_sk
           AND d_year = 1999
           AND d_moy = 12
           AND ( ( cd_marital_status = 'M'
                   AND cd_education_status = 'Unknown' )
                  OR ( cd_marital_status = 'W'
                       AND cd_education_status = 'Advanced Degree' ) )
           AND hd_buy_potential LIKE 'Unknown%'
           AND ca_gmt_offset = -7
    GROUP  BY cc_call_center_id,
              cc_name,
              cc_manager,
              cd_marital_status,
              cd_education_status
    ORDER  BY Sum(cr_net_loss) DESC;
    """


def polars_impl(run_config: RunConfig) -> pl.LazyFrame:
    """Query 91."""
    call_center = get_data(run_config.dataset_path, "call_center", run_config.suffix)
    catalog_returns = get_data(
        run_config.dataset_path, "catalog_returns", run_config.suffix
    )
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
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
    demo_filter = (
        (pl.col("cd_marital_status") == "M")
        & (pl.col("cd_education_status") == "Unknown")
    ) | (
        (pl.col("cd_marital_status") == "W")
        & (pl.col("cd_education_status") == "Advanced Degree")
    )
    return (
        catalog_returns.join(
            call_center,
            left_on="cr_call_center_sk",
            right_on="cc_call_center_sk",
            how="inner",
        )
        .join(
            date_dim, left_on="cr_returned_date_sk", right_on="d_date_sk", how="inner"
        )
        .join(
            customer,
            left_on="cr_returning_customer_sk",
            right_on="c_customer_sk",
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
            customer_address,
            left_on="c_current_addr_sk",
            right_on="ca_address_sk",
            how="inner",
        )
        .filter(
            (pl.col("d_year") == 1999)
            & (pl.col("d_moy") == 12)
            & demo_filter
            & (pl.col("hd_buy_potential").str.starts_with("Unknown"))
            & (pl.col("ca_gmt_offset") == -7)
        )
        .group_by(
            [
                "cc_call_center_id",
                "cc_name",
                "cc_manager",
                "cd_marital_status",
                "cd_education_status",
            ]
        )
        .agg([pl.col("cr_net_loss").sum().alias("sum(cr_net_loss)")])
        .select(
            [
                pl.col("cc_call_center_id").alias("Call_Center"),
                pl.col("cc_name").alias("Call_Center_Name"),
                pl.col("cc_manager").alias("Manager"),
                pl.col("sum(cr_net_loss)").alias("Returns_Loss"),
            ]
        )
        .sort("Returns_Loss", descending=True, nulls_last=True)
    )
