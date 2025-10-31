# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""Query 24."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.utils import get_data

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 24."""
    return """
        WITH ssales AS
        (SELECT c_last_name,
                c_first_name,
                s_store_name,
                ca_state,
                s_state,
                i_color,
                i_current_price,
                i_manager_id,
                i_units,
                i_size,
                sum(ss_net_paid) netpaid
        FROM store_sales,
                store_returns,
                store,
                item,
                customer,
                customer_address
        WHERE ss_ticket_number = sr_ticket_number
            AND ss_item_sk = sr_item_sk
            AND ss_customer_sk = c_customer_sk
            AND ss_item_sk = i_item_sk
            AND ss_store_sk = s_store_sk
            AND c_current_addr_sk = ca_address_sk
            AND c_birth_country <> upper(ca_country)
            AND s_zip = ca_zip
            AND s_market_id=8
        GROUP BY c_last_name,
                    c_first_name,
                    s_store_name,
                    ca_state,
                    s_state,
                    i_color,
                    i_current_price,
                    i_manager_id,
                    i_units,
                    i_size)
        SELECT c_last_name,
            c_first_name,
            s_store_name,
            sum(netpaid) paid
        FROM ssales
        WHERE i_color = 'peach'
        GROUP BY c_last_name,
                c_first_name,
                s_store_name
        HAVING sum(netpaid) >
        (SELECT 0.05*avg(netpaid)
        FROM ssales)
        ORDER BY c_last_name,
                c_first_name,
                s_store_name ;
    """


def polars_impl(run_config: RunConfig) -> pl.LazyFrame:
    """Query 24."""
    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    store_returns = get_data(
        run_config.dataset_path, "store_returns", run_config.suffix
    )
    store = get_data(run_config.dataset_path, "store", run_config.suffix)
    item = get_data(run_config.dataset_path, "item", run_config.suffix)
    customer = get_data(run_config.dataset_path, "customer", run_config.suffix)
    customer_address = get_data(
        run_config.dataset_path, "customer_address", run_config.suffix
    )

    ssales = (
        store_sales.join(
            store_returns,
            left_on=["ss_ticket_number", "ss_item_sk"],
            right_on=["sr_ticket_number", "sr_item_sk"],
        )
        .join(store, left_on="ss_store_sk", right_on="s_store_sk")
        .join(item, left_on="ss_item_sk", right_on="i_item_sk")
        .join(customer, left_on="ss_customer_sk", right_on="c_customer_sk")
        .join(customer_address, left_on="c_current_addr_sk", right_on="ca_address_sk")
        .filter(
            (pl.col("c_birth_country") != pl.col("ca_country").str.to_uppercase())
            & (pl.col("s_zip") == pl.col("ca_zip"))
            & (pl.col("s_market_id") == 8)
        )
        .group_by(
            [
                "c_last_name",
                "c_first_name",
                "s_store_name",
                "ca_state",
                "s_state",
                "i_color",
                "i_current_price",
                "i_manager_id",
                "i_units",
                "i_size",
            ]
        )
        .agg(pl.col("ss_net_paid").sum().alias("netpaid"))
    )

    threshold_table = ssales.select(
        (pl.col("netpaid").mean() * 0.05).alias("threshold")
    )

    return (
        ssales.filter(pl.col("i_color") == "peach")
        .group_by(["c_last_name", "c_first_name", "s_store_name"])
        .agg(pl.col("netpaid").sum().alias("paid"))
        .join(threshold_table, how="cross")
        .filter(pl.col("paid") > pl.col("threshold"))
        .select(["c_last_name", "c_first_name", "s_store_name", "paid"])
        .sort(["c_last_name", "c_first_name", "s_store_name"], nulls_last=True)
    )
