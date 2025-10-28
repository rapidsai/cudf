# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 66."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.utils import get_data

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 66."""
    return """
    SELECT w_warehouse_name, 
                   w_warehouse_sq_ft, 
                   w_city, 
                   w_county, 
                   w_state, 
                   w_country, 
                   ship_carriers, 
                   year1,
                   Sum(jan_sales)                     AS jan_sales, 
                   Sum(feb_sales)                     AS feb_sales, 
                   Sum(mar_sales)                     AS mar_sales, 
                   Sum(apr_sales)                     AS apr_sales, 
                   Sum(may_sales)                     AS may_sales, 
                   Sum(jun_sales)                     AS jun_sales, 
                   Sum(jul_sales)                     AS jul_sales, 
                   Sum(aug_sales)                     AS aug_sales, 
                   Sum(sep_sales)                     AS sep_sales, 
                   Sum(oct_sales)                     AS oct_sales, 
                   Sum(nov_sales)                     AS nov_sales, 
                   Sum(dec_sales)                     AS dec_sales, 
                   Sum(jan_sales / w_warehouse_sq_ft) AS jan_sales_per_sq_foot, 
                   Sum(feb_sales / w_warehouse_sq_ft) AS feb_sales_per_sq_foot, 
                   Sum(mar_sales / w_warehouse_sq_ft) AS mar_sales_per_sq_foot, 
                   Sum(apr_sales / w_warehouse_sq_ft) AS apr_sales_per_sq_foot, 
                   Sum(may_sales / w_warehouse_sq_ft) AS may_sales_per_sq_foot, 
                   Sum(jun_sales / w_warehouse_sq_ft) AS jun_sales_per_sq_foot, 
                   Sum(jul_sales / w_warehouse_sq_ft) AS jul_sales_per_sq_foot, 
                   Sum(aug_sales / w_warehouse_sq_ft) AS aug_sales_per_sq_foot, 
                   Sum(sep_sales / w_warehouse_sq_ft) AS sep_sales_per_sq_foot, 
                   Sum(oct_sales / w_warehouse_sq_ft) AS oct_sales_per_sq_foot, 
                   Sum(nov_sales / w_warehouse_sq_ft) AS nov_sales_per_sq_foot, 
                   Sum(dec_sales / w_warehouse_sq_ft) AS dec_sales_per_sq_foot, 
                   Sum(jan_net)                       AS jan_net, 
                   Sum(feb_net)                       AS feb_net, 
                   Sum(mar_net)                       AS mar_net, 
                   Sum(apr_net)                       AS apr_net, 
                   Sum(may_net)                       AS may_net, 
                   Sum(jun_net)                       AS jun_net, 
                   Sum(jul_net)                       AS jul_net, 
                   Sum(aug_net)                       AS aug_net, 
                   Sum(sep_net)                       AS sep_net, 
                   Sum(oct_net)                       AS oct_net, 
                   Sum(nov_net)                       AS nov_net, 
                   Sum(dec_net)                       AS dec_net 
    FROM   (SELECT w_warehouse_name, 
                   w_warehouse_sq_ft, 
                   w_city, 
                   w_county, 
                   w_state, 
                   w_country, 
                   'ZOUROS' 
                   || ',' 
                   || 'ZHOU' AS ship_carriers, 
                   d_year    AS year1, 
                   Sum(CASE 
                         WHEN d_moy = 1 THEN ws_ext_sales_price * ws_quantity 
                         ELSE 0 
                       END)  AS jan_sales, 
                   Sum(CASE 
                         WHEN d_moy = 2 THEN ws_ext_sales_price * ws_quantity 
                         ELSE 0 
                       END)  AS feb_sales, 
                   Sum(CASE 
                         WHEN d_moy = 3 THEN ws_ext_sales_price * ws_quantity 
                         ELSE 0 
                       END)  AS mar_sales, 
                   Sum(CASE 
                         WHEN d_moy = 4 THEN ws_ext_sales_price * ws_quantity 
                         ELSE 0 
                       END)  AS apr_sales, 
                   Sum(CASE 
                         WHEN d_moy = 5 THEN ws_ext_sales_price * ws_quantity 
                         ELSE 0 
                       END)  AS may_sales, 
                   Sum(CASE 
                         WHEN d_moy = 6 THEN ws_ext_sales_price * ws_quantity 
                         ELSE 0 
                       END)  AS jun_sales, 
                   Sum(CASE 
                         WHEN d_moy = 7 THEN ws_ext_sales_price * ws_quantity 
                         ELSE 0 
                       END)  AS jul_sales, 
                   Sum(CASE 
                         WHEN d_moy = 8 THEN ws_ext_sales_price * ws_quantity 
                         ELSE 0 
                       END)  AS aug_sales, 
                   Sum(CASE 
                         WHEN d_moy = 9 THEN ws_ext_sales_price * ws_quantity 
                         ELSE 0 
                       END)  AS sep_sales, 
                   Sum(CASE 
                         WHEN d_moy = 10 THEN ws_ext_sales_price * ws_quantity 
                         ELSE 0 
                       END)  AS oct_sales, 
                   Sum(CASE 
                         WHEN d_moy = 11 THEN ws_ext_sales_price * ws_quantity 
                         ELSE 0 
                       END)  AS nov_sales, 
                   Sum(CASE 
                         WHEN d_moy = 12 THEN ws_ext_sales_price * ws_quantity 
                         ELSE 0 
                       END)  AS dec_sales, 
                   Sum(CASE 
                         WHEN d_moy = 1 THEN ws_net_paid_inc_ship * ws_quantity 
                         ELSE 0 
                       END)  AS jan_net, 
                   Sum(CASE 
                         WHEN d_moy = 2 THEN ws_net_paid_inc_ship * ws_quantity 
                         ELSE 0 
                       END)  AS feb_net, 
                   Sum(CASE 
                         WHEN d_moy = 3 THEN ws_net_paid_inc_ship * ws_quantity 
                         ELSE 0 
                       END)  AS mar_net, 
                   Sum(CASE 
                         WHEN d_moy = 4 THEN ws_net_paid_inc_ship * ws_quantity 
                         ELSE 0 
                       END)  AS apr_net, 
                   Sum(CASE 
                         WHEN d_moy = 5 THEN ws_net_paid_inc_ship * ws_quantity 
                         ELSE 0 
                       END)  AS may_net, 
                   Sum(CASE 
                         WHEN d_moy = 6 THEN ws_net_paid_inc_ship * ws_quantity 
                         ELSE 0 
                       END)  AS jun_net, 
                   Sum(CASE 
                         WHEN d_moy = 7 THEN ws_net_paid_inc_ship * ws_quantity 
                         ELSE 0 
                       END)  AS jul_net, 
                   Sum(CASE 
                         WHEN d_moy = 8 THEN ws_net_paid_inc_ship * ws_quantity 
                         ELSE 0 
                       END)  AS aug_net, 
                   Sum(CASE 
                         WHEN d_moy = 9 THEN ws_net_paid_inc_ship * ws_quantity 
                         ELSE 0 
                       END)  AS sep_net, 
                   Sum(CASE 
                         WHEN d_moy = 10 THEN ws_net_paid_inc_ship * ws_quantity 
                         ELSE 0 
                       END)  AS oct_net, 
                   Sum(CASE 
                         WHEN d_moy = 11 THEN ws_net_paid_inc_ship * ws_quantity 
                         ELSE 0 
                       END)  AS nov_net, 
                   Sum(CASE 
                         WHEN d_moy = 12 THEN ws_net_paid_inc_ship * ws_quantity 
                         ELSE 0 
                       END)  AS dec_net 
            FROM   web_sales, 
                   warehouse, 
                   date_dim, 
                   time_dim, 
                   ship_mode 
            WHERE  ws_warehouse_sk = w_warehouse_sk 
                   AND ws_sold_date_sk = d_date_sk 
                   AND ws_sold_time_sk = t_time_sk 
                   AND ws_ship_mode_sk = sm_ship_mode_sk 
                   AND d_year = 1998 
                   AND t_time BETWEEN 7249 AND 7249 + 28800 
                   AND sm_carrier IN ( 'ZOUROS', 'ZHOU' ) 
            GROUP  BY w_warehouse_name, 
                      w_warehouse_sq_ft, 
                      w_city, 
                      w_county, 
                      w_state, 
                      w_country, 
                      d_year 
            UNION ALL 
            SELECT w_warehouse_name, 
                   w_warehouse_sq_ft, 
                   w_city, 
                   w_county, 
                   w_state, 
                   w_country, 
                   'ZOUROS' 
                   || ',' 
                   || 'ZHOU' AS ship_carriers, 
                   d_year    AS year1, 
                   Sum(CASE 
                         WHEN d_moy = 1 THEN cs_ext_sales_price * cs_quantity 
                         ELSE 0 
                       END)  AS jan_sales, 
                   Sum(CASE 
                         WHEN d_moy = 2 THEN cs_ext_sales_price * cs_quantity 
                         ELSE 0 
                       END)  AS feb_sales, 
                   Sum(CASE 
                         WHEN d_moy = 3 THEN cs_ext_sales_price * cs_quantity 
                         ELSE 0 
                       END)  AS mar_sales, 
                   Sum(CASE 
                         WHEN d_moy = 4 THEN cs_ext_sales_price * cs_quantity 
                         ELSE 0 
                       END)  AS apr_sales, 
                   Sum(CASE 
                         WHEN d_moy = 5 THEN cs_ext_sales_price * cs_quantity 
                         ELSE 0 
                       END)  AS may_sales, 
                   Sum(CASE 
                         WHEN d_moy = 6 THEN cs_ext_sales_price * cs_quantity 
                         ELSE 0 
                       END)  AS jun_sales, 
                   Sum(CASE 
                         WHEN d_moy = 7 THEN cs_ext_sales_price * cs_quantity 
                         ELSE 0 
                       END)  AS jul_sales, 
                   Sum(CASE 
                         WHEN d_moy = 8 THEN cs_ext_sales_price * cs_quantity 
                         ELSE 0 
                       END)  AS aug_sales, 
                   Sum(CASE 
                         WHEN d_moy = 9 THEN cs_ext_sales_price * cs_quantity 
                         ELSE 0 
                       END)  AS sep_sales, 
                   Sum(CASE 
                         WHEN d_moy = 10 THEN cs_ext_sales_price * cs_quantity 
                         ELSE 0 
                       END)  AS oct_sales, 
                   Sum(CASE 
                         WHEN d_moy = 11 THEN cs_ext_sales_price * cs_quantity 
                         ELSE 0 
                       END)  AS nov_sales, 
                   Sum(CASE 
                         WHEN d_moy = 12 THEN cs_ext_sales_price * cs_quantity 
                         ELSE 0 
                       END)  AS dec_sales, 
                   Sum(CASE 
                         WHEN d_moy = 1 THEN cs_net_paid * cs_quantity 
                         ELSE 0 
                       END)  AS jan_net, 
                   Sum(CASE 
                         WHEN d_moy = 2 THEN cs_net_paid * cs_quantity 
                         ELSE 0 
                       END)  AS feb_net, 
                   Sum(CASE 
                         WHEN d_moy = 3 THEN cs_net_paid * cs_quantity 
                         ELSE 0 
                       END)  AS mar_net, 
                   Sum(CASE 
                         WHEN d_moy = 4 THEN cs_net_paid * cs_quantity 
                         ELSE 0 
                       END)  AS apr_net, 
                   Sum(CASE 
                         WHEN d_moy = 5 THEN cs_net_paid * cs_quantity 
                         ELSE 0 
                       END)  AS may_net, 
                   Sum(CASE 
                         WHEN d_moy = 6 THEN cs_net_paid * cs_quantity 
                         ELSE 0 
                       END)  AS jun_net, 
                   Sum(CASE 
                         WHEN d_moy = 7 THEN cs_net_paid * cs_quantity 
                         ELSE 0 
                       END)  AS jul_net, 
                   Sum(CASE 
                         WHEN d_moy = 8 THEN cs_net_paid * cs_quantity 
                         ELSE 0 
                       END)  AS aug_net, 
                   Sum(CASE 
                         WHEN d_moy = 9 THEN cs_net_paid * cs_quantity 
                         ELSE 0 
                       END)  AS sep_net, 
                   Sum(CASE 
                         WHEN d_moy = 10 THEN cs_net_paid * cs_quantity 
                         ELSE 0 
                       END)  AS oct_net, 
                   Sum(CASE 
                         WHEN d_moy = 11 THEN cs_net_paid * cs_quantity 
                         ELSE 0 
                       END)  AS nov_net, 
                   Sum(CASE 
                         WHEN d_moy = 12 THEN cs_net_paid * cs_quantity 
                         ELSE 0 
                       END)  AS dec_net 
            FROM   catalog_sales, 
                   warehouse, 
                   date_dim, 
                   time_dim, 
                   ship_mode 
            WHERE  cs_warehouse_sk = w_warehouse_sk 
                   AND cs_sold_date_sk = d_date_sk 
                   AND cs_sold_time_sk = t_time_sk 
                   AND cs_ship_mode_sk = sm_ship_mode_sk 
                   AND d_year = 1998 
                   AND t_time BETWEEN 7249 AND 7249 + 28800 
                   AND sm_carrier IN ( 'ZOUROS', 'ZHOU' ) 
            GROUP  BY w_warehouse_name, 
                      w_warehouse_sq_ft, 
                      w_city, 
                      w_county, 
                      w_state, 
                      w_country, 
                      d_year) x 
    GROUP  BY w_warehouse_name, 
              w_warehouse_sq_ft, 
              w_city, 
              w_county, 
              w_state, 
              w_country, 
              ship_carriers, 
              year1 
    ORDER  BY w_warehouse_name
    LIMIT 100;
    """


def polars_impl(run_config: RunConfig) -> pl.LazyFrame:
    """Query 66."""
    web_sales = get_data(run_config.dataset_path, "web_sales", run_config.suffix)
    catalog_sales = get_data(run_config.dataset_path, "catalog_sales", run_config.suffix)
    warehouse = get_data(run_config.dataset_path, "warehouse", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    time_dim = get_data(run_config.dataset_path, "time_dim", run_config.suffix)
    ship_mode = get_data(run_config.dataset_path, "ship_mode", run_config.suffix)

    month_names = ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]

    web_base = (
        web_sales
        .join(warehouse, left_on="ws_warehouse_sk", right_on="w_warehouse_sk")
        .join(date_dim, left_on="ws_sold_date_sk", right_on="d_date_sk")
        .join(time_dim, left_on="ws_sold_time_sk", right_on="t_time_sk")
        .join(ship_mode, left_on="ws_ship_mode_sk", right_on="sm_ship_mode_sk")
        .filter(
            (pl.col("d_year") == 1998)
            & pl.col("t_time").is_between(7249, 7249 + 28800)
            & pl.col("sm_carrier").is_in(["ZOUROS", "ZHOU"])
        )
    )

    web_sales_aggs = [
        pl.when(pl.col("d_moy") == (i+1)).then(pl.col("ws_ext_sales_price") * pl.col("ws_quantity")).otherwise(0).sum().alias(f"{month_names[i]}_sales")
        for i in range(12)
    ]
    web_net_aggs = [
        pl.when(pl.col("d_moy") == (i+1)).then(pl.col("ws_net_paid_inc_ship") * pl.col("ws_quantity")).otherwise(0).sum().alias(f"{month_names[i]}_net")
        for i in range(12)
    ]
    web_sales_monthly = (
        web_base
        .group_by(["w_warehouse_name", "w_warehouse_sq_ft", "w_city", "w_county", "w_state", "w_country", "d_year"])
        .agg(web_sales_aggs + web_net_aggs)
        .with_columns([pl.lit("ZOUROS,ZHOU").alias("ship_carriers"), pl.col("d_year").alias("year1")])
    )

    cat_base = (
        catalog_sales
        .join(warehouse, left_on="cs_warehouse_sk", right_on="w_warehouse_sk")
        .join(date_dim, left_on="cs_sold_date_sk", right_on="d_date_sk")
        .join(time_dim, left_on="cs_sold_time_sk", right_on="t_time_sk")
        .join(ship_mode, left_on="cs_ship_mode_sk", right_on="sm_ship_mode_sk")
        .filter(
            (pl.col("d_year") == 1998)
            & pl.col("t_time").is_between(7249, 7249 + 28800)
            & pl.col("sm_carrier").is_in(["ZOUROS", "ZHOU"])
        )
    )

    cat_sales_aggs = [
        pl.when(pl.col("d_moy") == (i+1)).then(pl.col("cs_ext_sales_price") * pl.col("cs_quantity")).otherwise(0).sum().alias(f"{month_names[i]}_sales")
        for i in range(12)
    ]
    cat_net_aggs = [
        pl.when(pl.col("d_moy") == (i+1)).then(pl.col("cs_net_paid") * pl.col("cs_quantity")).otherwise(0).sum().alias(f"{month_names[i]}_net")
        for i in range(12)
    ]
    catalog_sales_monthly = (
        cat_base
        .group_by(["w_warehouse_name", "w_warehouse_sq_ft", "w_city", "w_county", "w_state", "w_country", "d_year"])
        .agg(cat_sales_aggs + cat_net_aggs)
        .with_columns([pl.lit("ZOUROS,ZHOU").alias("ship_carriers"), pl.col("d_year").alias("year1")])
    )

    per_sqft_exprs = [
        (pl.col(f"{m}_sales") / pl.col("w_warehouse_sq_ft")).alias(f"{m}_sales_per_sq_foot")
        for m in month_names
    ]

    return (
        pl.concat([web_sales_monthly, catalog_sales_monthly])
        .group_by(["w_warehouse_name", "w_warehouse_sq_ft", "w_city", "w_county", "w_state", "w_country", "ship_carriers", "year1"])
        .agg(
            [pl.col(f"{m}_sales").sum().alias(f"{m}_sales") for m in month_names]
            + [pl.col(f"{m}_net").sum().alias(f"{m}_net") for m in month_names]
        )
        .with_columns(per_sqft_exprs)
        .sort(["w_warehouse_name"], nulls_last=True, descending=[False])
        .limit(100)
    )
