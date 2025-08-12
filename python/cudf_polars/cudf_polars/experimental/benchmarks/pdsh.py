# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""
Experimental PDS-H benchmarks.

Based on https://github.com/pola-rs/polars-benchmark.

WARNING: This is an experimental (and unofficial)
benchmark script. It is not intended for public use
and may be modified or removed at any time.
"""

from __future__ import annotations

import contextlib
import os
from datetime import date
from typing import TYPE_CHECKING

import polars as pl

with contextlib.suppress(ImportError):
    from cudf_polars.experimental.benchmarks.utils import (
        get_data,
        run_polars,
    )


if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig

# Without this setting, the first IO task to run
# on each worker takes ~15 sec extra
os.environ["KVIKIO_COMPAT_MODE"] = os.environ.get("KVIKIO_COMPAT_MODE", "on")
os.environ["KVIKIO_NTHREADS"] = os.environ.get("KVIKIO_NTHREADS", "8")


class PDSHQueries:
    """PDS-H query definitions."""

    name: str = "pdsh"

    @staticmethod
    def q0(run_config: RunConfig) -> pl.LazyFrame:
        """Query 0."""
        return pl.LazyFrame()

    @staticmethod
    def q1(run_config: RunConfig) -> pl.LazyFrame:
        """Query 1."""
        lineitem = get_data(run_config.dataset_path, "lineitem", run_config.suffix)

        var1 = date(1998, 9, 2)

        return (
            lineitem.filter(pl.col("l_shipdate") <= var1)
            .group_by("l_returnflag", "l_linestatus")
            .agg(
                pl.sum("l_quantity").alias("sum_qty"),
                pl.sum("l_extendedprice").alias("sum_base_price"),
                (pl.col("l_extendedprice") * (1.0 - pl.col("l_discount")))
                .sum()
                .alias("sum_disc_price"),
                (
                    pl.col("l_extendedprice")
                    * (1.0 - pl.col("l_discount"))
                    * (1.0 + pl.col("l_tax"))
                )
                .sum()
                .alias("sum_charge"),
                pl.mean("l_quantity").alias("avg_qty"),
                pl.mean("l_extendedprice").alias("avg_price"),
                pl.mean("l_discount").alias("avg_disc"),
                pl.len().alias("count_order"),
            )
            .sort("l_returnflag", "l_linestatus")
        )

    @staticmethod
    def q2(run_config: RunConfig) -> pl.LazyFrame:
        """Query 2."""
        nation = get_data(run_config.dataset_path, "nation", run_config.suffix)
        part = get_data(run_config.dataset_path, "part", run_config.suffix)
        partsupp = get_data(run_config.dataset_path, "partsupp", run_config.suffix)
        region = get_data(run_config.dataset_path, "region", run_config.suffix)
        supplier = get_data(run_config.dataset_path, "supplier", run_config.suffix)

        var1 = 15
        var2 = "BRASS"
        var3 = "EUROPE"

        q1 = (
            part.join(partsupp, left_on="p_partkey", right_on="ps_partkey")
            .join(supplier, left_on="ps_suppkey", right_on="s_suppkey")
            .join(nation, left_on="s_nationkey", right_on="n_nationkey")
            .join(region, left_on="n_regionkey", right_on="r_regionkey")
            .filter(pl.col("p_size") == var1)
            .filter(pl.col("p_type").str.ends_with(var2))
            .filter(pl.col("r_name") == var3)
        )

        return (
            q1.group_by("p_partkey")
            .agg(pl.min("ps_supplycost"))
            .join(q1, on=["p_partkey", "ps_supplycost"])
            .select(
                "s_acctbal",
                "s_name",
                "n_name",
                "p_partkey",
                "p_mfgr",
                "s_address",
                "s_phone",
                "s_comment",
            )
            .sort(
                by=["s_acctbal", "n_name", "s_name", "p_partkey"],
                descending=[True, False, False, False],
            )
            .head(100)
        )

    @staticmethod
    def q3(run_config: RunConfig) -> pl.LazyFrame:
        """Query 3."""
        customer = get_data(run_config.dataset_path, "customer", run_config.suffix)
        lineitem = get_data(run_config.dataset_path, "lineitem", run_config.suffix)
        orders = get_data(run_config.dataset_path, "orders", run_config.suffix)

        var1 = "BUILDING"
        var2 = date(1995, 3, 15)

        return (
            customer.filter(pl.col("c_mktsegment") == var1)
            .join(orders, left_on="c_custkey", right_on="o_custkey")
            .join(lineitem, left_on="o_orderkey", right_on="l_orderkey")
            .filter(pl.col("o_orderdate") < var2)
            .filter(pl.col("l_shipdate") > var2)
            .with_columns(
                (pl.col("l_extendedprice") * (1 - pl.col("l_discount"))).alias(
                    "revenue"
                )
            )
            .group_by("o_orderkey", "o_orderdate", "o_shippriority")
            .agg(pl.sum("revenue"))
            .select(
                pl.col("o_orderkey").alias("l_orderkey"),
                "revenue",
                "o_orderdate",
                "o_shippriority",
            )
            .sort(by=["revenue", "o_orderdate"], descending=[True, False])
            .head(10)
        )

    @staticmethod
    def q4(run_config: RunConfig) -> pl.LazyFrame:
        """Query 4."""
        lineitem = get_data(run_config.dataset_path, "lineitem", run_config.suffix)
        orders = get_data(run_config.dataset_path, "orders", run_config.suffix)

        var1 = date(1993, 7, 1)
        var2 = date(1993, 10, 1)

        return (
            # SQL exists translates to semi join in Polars API
            orders.join(
                (lineitem.filter(pl.col("l_commitdate") < pl.col("l_receiptdate"))),
                left_on="o_orderkey",
                right_on="l_orderkey",
                how="semi",
            )
            .filter(pl.col("o_orderdate").is_between(var1, var2, closed="left"))
            .group_by("o_orderpriority")
            .agg(pl.len().alias("order_count"))
            .sort("o_orderpriority")
        )

    @staticmethod
    def q5(run_config: RunConfig) -> pl.LazyFrame:
        """Query 5."""
        path = run_config.dataset_path
        suffix = run_config.suffix
        customer = get_data(path, "customer", suffix)
        lineitem = get_data(path, "lineitem", suffix)
        nation = get_data(path, "nation", suffix)
        orders = get_data(path, "orders", suffix)
        region = get_data(path, "region", suffix)
        supplier = get_data(path, "supplier", suffix)

        var1 = "ASIA"
        var2 = date(1994, 1, 1)
        var3 = date(1995, 1, 1)

        return (
            region.join(nation, left_on="r_regionkey", right_on="n_regionkey")
            .join(customer, left_on="n_nationkey", right_on="c_nationkey")
            .join(orders, left_on="c_custkey", right_on="o_custkey")
            .join(lineitem, left_on="o_orderkey", right_on="l_orderkey")
            .join(
                supplier,
                left_on=["l_suppkey", "n_nationkey"],
                right_on=["s_suppkey", "s_nationkey"],
            )
            .filter(pl.col("r_name") == var1)
            .filter(pl.col("o_orderdate").is_between(var2, var3, closed="left"))
            .with_columns(
                (pl.col("l_extendedprice") * (1 - pl.col("l_discount"))).alias(
                    "revenue"
                )
            )
            .group_by("n_name")
            .agg(pl.sum("revenue"))
            .sort(by="revenue", descending=True)
        )

    @staticmethod
    def q6(run_config: RunConfig) -> pl.LazyFrame:
        """Query 6."""
        path = run_config.dataset_path
        suffix = run_config.suffix
        lineitem = get_data(path, "lineitem", suffix)

        var1 = date(1994, 1, 1)
        var2 = date(1995, 1, 1)
        var3 = 0.05
        var4 = 0.07
        var5 = 24

        return (
            lineitem.filter(pl.col("l_shipdate").is_between(var1, var2, closed="left"))
            .filter(pl.col("l_discount").is_between(var3, var4))
            .filter(pl.col("l_quantity") < var5)
            .with_columns(
                (pl.col("l_extendedprice") * pl.col("l_discount")).alias("revenue")
            )
            .select(pl.sum("revenue"))
        )

    @staticmethod
    def q7(run_config: RunConfig) -> pl.LazyFrame:
        """Query 7."""
        customer = get_data(run_config.dataset_path, "customer", run_config.suffix)
        lineitem = get_data(run_config.dataset_path, "lineitem", run_config.suffix)
        nation = get_data(run_config.dataset_path, "nation", run_config.suffix)
        orders = get_data(run_config.dataset_path, "orders", run_config.suffix)
        supplier = get_data(run_config.dataset_path, "supplier", run_config.suffix)

        var1 = "FRANCE"
        var2 = "GERMANY"
        var3 = date(1995, 1, 1)
        var4 = date(1996, 12, 31)

        n1 = nation.filter(pl.col("n_name") == var1)
        n2 = nation.filter(pl.col("n_name") == var2)

        q1 = (
            customer.join(n1, left_on="c_nationkey", right_on="n_nationkey")
            .join(orders, left_on="c_custkey", right_on="o_custkey")
            .rename({"n_name": "cust_nation"})
            .join(lineitem, left_on="o_orderkey", right_on="l_orderkey")
            .join(supplier, left_on="l_suppkey", right_on="s_suppkey")
            .join(n2, left_on="s_nationkey", right_on="n_nationkey")
            .rename({"n_name": "supp_nation"})
        )

        q2 = (
            customer.join(n2, left_on="c_nationkey", right_on="n_nationkey")
            .join(orders, left_on="c_custkey", right_on="o_custkey")
            .rename({"n_name": "cust_nation"})
            .join(lineitem, left_on="o_orderkey", right_on="l_orderkey")
            .join(supplier, left_on="l_suppkey", right_on="s_suppkey")
            .join(n1, left_on="s_nationkey", right_on="n_nationkey")
            .rename({"n_name": "supp_nation"})
        )

        return (
            pl.concat([q1, q2])
            .filter(pl.col("l_shipdate").is_between(var3, var4))
            .with_columns(
                (pl.col("l_extendedprice") * (1 - pl.col("l_discount"))).alias(
                    "volume"
                ),
                pl.col("l_shipdate").dt.year().alias("l_year"),
            )
            .group_by("supp_nation", "cust_nation", "l_year")
            .agg(pl.sum("volume").alias("revenue"))
            .sort(by=["supp_nation", "cust_nation", "l_year"])
        )

    @staticmethod
    def q8(run_config: RunConfig) -> pl.LazyFrame:
        """Query 8."""
        customer = get_data(run_config.dataset_path, "customer", run_config.suffix)
        lineitem = get_data(run_config.dataset_path, "lineitem", run_config.suffix)
        nation = get_data(run_config.dataset_path, "nation", run_config.suffix)
        orders = get_data(run_config.dataset_path, "orders", run_config.suffix)
        part = get_data(run_config.dataset_path, "part", run_config.suffix)
        region = get_data(run_config.dataset_path, "region", run_config.suffix)
        supplier = get_data(run_config.dataset_path, "supplier", run_config.suffix)

        var1 = "BRAZIL"
        var2 = "AMERICA"
        var3 = "ECONOMY ANODIZED STEEL"
        var4 = date(1995, 1, 1)
        var5 = date(1996, 12, 31)

        n1 = nation.select("n_nationkey", "n_regionkey")
        n2 = nation.select("n_nationkey", "n_name")

        return (
            part.join(lineitem, left_on="p_partkey", right_on="l_partkey")
            .join(supplier, left_on="l_suppkey", right_on="s_suppkey")
            .join(orders, left_on="l_orderkey", right_on="o_orderkey")
            .join(customer, left_on="o_custkey", right_on="c_custkey")
            .join(n1, left_on="c_nationkey", right_on="n_nationkey")
            .join(region, left_on="n_regionkey", right_on="r_regionkey")
            .filter(pl.col("r_name") == var2)
            .join(n2, left_on="s_nationkey", right_on="n_nationkey")
            .filter(pl.col("o_orderdate").is_between(var4, var5))
            .filter(pl.col("p_type") == var3)
            .select(
                pl.col("o_orderdate").dt.year().alias("o_year"),
                (pl.col("l_extendedprice") * (1 - pl.col("l_discount"))).alias(
                    "volume"
                ),
                pl.col("n_name").alias("nation"),
            )
            .with_columns(
                pl.when(pl.col("nation") == var1)
                .then(pl.col("volume"))
                .otherwise(0)
                .alias("_tmp")
            )
            .group_by("o_year")
            .agg((pl.sum("_tmp") / pl.sum("volume")).round(2).alias("mkt_share"))
            .sort("o_year")
        )

    @staticmethod
    def q9(run_config: RunConfig) -> pl.LazyFrame:
        """Query 9."""
        path = run_config.dataset_path
        suffix = run_config.suffix
        lineitem = get_data(path, "lineitem", suffix)
        nation = get_data(path, "nation", suffix)
        orders = get_data(path, "orders", suffix)
        part = get_data(path, "part", suffix)
        partsupp = get_data(path, "partsupp", suffix)
        supplier = get_data(path, "supplier", suffix)

        return (
            part.join(partsupp, left_on="p_partkey", right_on="ps_partkey")
            .join(supplier, left_on="ps_suppkey", right_on="s_suppkey")
            .join(
                lineitem,
                left_on=["p_partkey", "ps_suppkey"],
                right_on=["l_partkey", "l_suppkey"],
            )
            .join(orders, left_on="l_orderkey", right_on="o_orderkey")
            .join(nation, left_on="s_nationkey", right_on="n_nationkey")
            .filter(pl.col("p_name").str.contains("green"))
            .select(
                pl.col("n_name").alias("nation"),
                pl.col("o_orderdate").dt.year().alias("o_year"),
                (
                    pl.col("l_extendedprice") * (1 - pl.col("l_discount"))
                    - pl.col("ps_supplycost") * pl.col("l_quantity")
                ).alias("amount"),
            )
            .group_by("nation", "o_year")
            .agg(pl.sum("amount").round(2).alias("sum_profit"))
            .sort(by=["nation", "o_year"], descending=[False, True])
        )

    @staticmethod
    def q10(run_config: RunConfig) -> pl.LazyFrame:
        """Query 10."""
        path = run_config.dataset_path
        suffix = run_config.suffix
        customer = get_data(path, "customer", suffix)
        lineitem = get_data(path, "lineitem", suffix)
        nation = get_data(path, "nation", suffix)
        orders = get_data(path, "orders", suffix)

        var1 = date(1993, 10, 1)
        var2 = date(1994, 1, 1)

        return (
            customer.join(orders, left_on="c_custkey", right_on="o_custkey")
            .join(lineitem, left_on="o_orderkey", right_on="l_orderkey")
            .join(nation, left_on="c_nationkey", right_on="n_nationkey")
            .filter(pl.col("o_orderdate").is_between(var1, var2, closed="left"))
            .filter(pl.col("l_returnflag") == "R")
            .group_by(
                "c_custkey",
                "c_name",
                "c_acctbal",
                "c_phone",
                "n_name",
                "c_address",
                "c_comment",
            )
            .agg(
                (pl.col("l_extendedprice") * (1 - pl.col("l_discount")))
                .sum()
                .round(2)
                .alias("revenue")
            )
            .select(
                "c_custkey",
                "c_name",
                "revenue",
                "c_acctbal",
                "n_name",
                "c_address",
                "c_phone",
                "c_comment",
            )
            .sort(by="revenue", descending=True)
            .head(20)
        )

    @staticmethod
    def q11(run_config: RunConfig) -> pl.LazyFrame:
        """Query 11."""
        nation = get_data(run_config.dataset_path, "nation", run_config.suffix)
        partsupp = get_data(run_config.dataset_path, "partsupp", run_config.suffix)
        supplier = get_data(run_config.dataset_path, "supplier", run_config.suffix)

        var1 = "GERMANY"
        var2 = 0.0001 / run_config.scale_factor

        q1 = (
            partsupp.join(supplier, left_on="ps_suppkey", right_on="s_suppkey")
            .join(nation, left_on="s_nationkey", right_on="n_nationkey")
            .filter(pl.col("n_name") == var1)
        )
        q2 = q1.select(
            (pl.col("ps_supplycost") * pl.col("ps_availqty"))
            .sum()
            .round(2)
            .alias("tmp")
            * var2
        )

        return (
            q1.group_by("ps_partkey")
            .agg(
                (pl.col("ps_supplycost") * pl.col("ps_availqty"))
                .sum()
                .round(2)
                .alias("value")
            )
            .join(q2, how="cross")
            .filter(pl.col("value") > pl.col("tmp"))
            .select("ps_partkey", "value")
            .sort("value", descending=True)
        )

    @staticmethod
    def q12(run_config: RunConfig) -> pl.LazyFrame:
        """Query 12."""
        lineitem = get_data(run_config.dataset_path, "lineitem", run_config.suffix)
        orders = get_data(run_config.dataset_path, "orders", run_config.suffix)

        var1 = "MAIL"
        var2 = "SHIP"
        var3 = date(1994, 1, 1)
        var4 = date(1995, 1, 1)

        return (
            orders.join(lineitem, left_on="o_orderkey", right_on="l_orderkey")
            .filter(pl.col("l_shipmode").is_in([var1, var2]))
            .filter(pl.col("l_commitdate") < pl.col("l_receiptdate"))
            .filter(pl.col("l_shipdate") < pl.col("l_commitdate"))
            .filter(pl.col("l_receiptdate").is_between(var3, var4, closed="left"))
            .with_columns(
                pl.when(pl.col("o_orderpriority").is_in(["1-URGENT", "2-HIGH"]))
                .then(1)
                .otherwise(0)
                .alias("high_line_count"),
                pl.when(pl.col("o_orderpriority").is_in(["1-URGENT", "2-HIGH"]).not_())
                .then(1)
                .otherwise(0)
                .alias("low_line_count"),
            )
            .group_by("l_shipmode")
            .agg(pl.col("high_line_count").sum(), pl.col("low_line_count").sum())
            .sort("l_shipmode")
        )

    @staticmethod
    def q13(run_config: RunConfig) -> pl.LazyFrame:
        """Query 13."""
        customer = get_data(run_config.dataset_path, "customer", run_config.suffix)
        orders = get_data(run_config.dataset_path, "orders", run_config.suffix)

        var1 = "special"
        var2 = "requests"

        orders = orders.filter(
            pl.col("o_comment").str.contains(f"{var1}.*{var2}").not_()
        )
        return (
            customer.join(orders, left_on="c_custkey", right_on="o_custkey", how="left")
            .group_by("c_custkey")
            .agg(pl.col("o_orderkey").count().alias("c_count"))
            .group_by("c_count")
            .len()
            .select(pl.col("c_count"), pl.col("len").alias("custdist"))
            .sort(by=["custdist", "c_count"], descending=[True, True])
        )

    @staticmethod
    def q14(run_config: RunConfig) -> pl.LazyFrame:
        """Query 14."""
        lineitem = get_data(run_config.dataset_path, "lineitem", run_config.suffix)
        part = get_data(run_config.dataset_path, "part", run_config.suffix)

        var1 = date(1995, 9, 1)
        var2 = date(1995, 10, 1)

        return (
            lineitem.join(part, left_on="l_partkey", right_on="p_partkey")
            .filter(pl.col("l_shipdate").is_between(var1, var2, closed="left"))
            .select(
                (
                    100.00
                    * pl.when(pl.col("p_type").str.contains("PROMO*"))
                    .then(pl.col("l_extendedprice") * (1 - pl.col("l_discount")))
                    .otherwise(0)
                    .sum()
                    / (pl.col("l_extendedprice") * (1 - pl.col("l_discount"))).sum()
                )
                .round(2)
                .alias("promo_revenue")
            )
        )

    @staticmethod
    def q15(run_config: RunConfig) -> pl.LazyFrame:
        """Query 15."""
        lineitem = get_data(run_config.dataset_path, "lineitem", run_config.suffix)
        supplier = get_data(run_config.dataset_path, "supplier", run_config.suffix)

        var1 = date(1996, 1, 1)
        var2 = date(1996, 4, 1)

        revenue = (
            lineitem.filter(pl.col("l_shipdate").is_between(var1, var2, closed="left"))
            .group_by("l_suppkey")
            .agg(
                (pl.col("l_extendedprice") * (1 - pl.col("l_discount")))
                .sum()
                .alias("total_revenue")
            )
            .select(pl.col("l_suppkey").alias("supplier_no"), pl.col("total_revenue"))
        )

        return (
            supplier.join(revenue, left_on="s_suppkey", right_on="supplier_no")
            .filter(pl.col("total_revenue") == pl.col("total_revenue").max())
            .with_columns(pl.col("total_revenue").round(2))
            .select("s_suppkey", "s_name", "s_address", "s_phone", "total_revenue")
            .sort("s_suppkey")
        )

    @staticmethod
    def q16(run_config: RunConfig) -> pl.LazyFrame:
        """Query 16."""
        part = get_data(run_config.dataset_path, "part", run_config.suffix)
        partsupp = get_data(run_config.dataset_path, "partsupp", run_config.suffix)
        supplier = get_data(run_config.dataset_path, "supplier", run_config.suffix)

        var1 = "Brand#45"

        supplier = supplier.filter(
            pl.col("s_comment").str.contains(".*Customer.*Complaints.*")
        ).select(pl.col("s_suppkey"), pl.col("s_suppkey").alias("ps_suppkey"))

        return (
            part.join(partsupp, left_on="p_partkey", right_on="ps_partkey")
            .filter(pl.col("p_brand") != var1)
            .filter(pl.col("p_type").str.contains("MEDIUM POLISHED*").not_())
            .filter(pl.col("p_size").is_in([49, 14, 23, 45, 19, 3, 36, 9]))
            .join(supplier, left_on="ps_suppkey", right_on="s_suppkey", how="left")
            .filter(pl.col("ps_suppkey_right").is_null())
            .group_by("p_brand", "p_type", "p_size")
            .agg(pl.col("ps_suppkey").n_unique().alias("supplier_cnt"))
            .sort(
                by=["supplier_cnt", "p_brand", "p_type", "p_size"],
                descending=[True, False, False, False],
            )
        )

    @staticmethod
    def q17(run_config: RunConfig) -> pl.LazyFrame:
        """Query 17."""
        lineitem = get_data(run_config.dataset_path, "lineitem", run_config.suffix)
        part = get_data(run_config.dataset_path, "part", run_config.suffix)

        var1 = "Brand#23"
        var2 = "MED BOX"

        q1 = (
            part.filter(pl.col("p_brand") == var1)
            .filter(pl.col("p_container") == var2)
            .join(lineitem, how="left", left_on="p_partkey", right_on="l_partkey")
        )

        return (
            q1.group_by("p_partkey")
            .agg((0.2 * pl.col("l_quantity").mean()).alias("avg_quantity"))
            .select(pl.col("p_partkey").alias("key"), pl.col("avg_quantity"))
            .join(q1, left_on="key", right_on="p_partkey")
            .filter(pl.col("l_quantity") < pl.col("avg_quantity"))
            .select(
                (pl.col("l_extendedprice").sum() / 7.0).round(2).alias("avg_yearly")
            )
        )

    @staticmethod
    def q18(run_config: RunConfig) -> pl.LazyFrame:
        """Query 18."""
        path = run_config.dataset_path
        suffix = run_config.suffix
        customer = get_data(path, "customer", suffix)
        lineitem = get_data(path, "lineitem", suffix)
        orders = get_data(path, "orders", suffix)

        var1 = 300

        q1 = (
            lineitem.group_by("l_orderkey")
            .agg(pl.col("l_quantity").sum().alias("sum_quantity"))
            .filter(pl.col("sum_quantity") > var1)
        )

        return (
            orders.join(q1, left_on="o_orderkey", right_on="l_orderkey", how="semi")
            .join(lineitem, left_on="o_orderkey", right_on="l_orderkey")
            .join(customer, left_on="o_custkey", right_on="c_custkey")
            .group_by(
                "c_name", "o_custkey", "o_orderkey", "o_orderdate", "o_totalprice"
            )
            .agg(pl.col("l_quantity").sum().alias("col6"))
            .select(
                pl.col("c_name"),
                pl.col("o_custkey").alias("c_custkey"),
                pl.col("o_orderkey"),
                pl.col("o_orderdate").alias("o_orderdat"),
                pl.col("o_totalprice"),
                pl.col("col6"),
            )
            .sort(by=["o_totalprice", "o_orderdat"], descending=[True, False])
            .head(100)
        )

    @staticmethod
    def q19(run_config: RunConfig) -> pl.LazyFrame:
        """Query 19."""
        lineitem = get_data(run_config.dataset_path, "lineitem", run_config.suffix)
        part = get_data(run_config.dataset_path, "part", run_config.suffix)

        return (
            part.join(lineitem, left_on="p_partkey", right_on="l_partkey")
            .filter(pl.col("l_shipmode").is_in(["AIR", "AIR REG"]))
            .filter(pl.col("l_shipinstruct") == "DELIVER IN PERSON")
            .filter(
                (
                    (pl.col("p_brand") == "Brand#12")
                    & pl.col("p_container").is_in(
                        ["SM CASE", "SM BOX", "SM PACK", "SM PKG"]
                    )
                    & (pl.col("l_quantity").is_between(1, 11))
                    & (pl.col("p_size").is_between(1, 5))
                )
                | (
                    (pl.col("p_brand") == "Brand#23")
                    & pl.col("p_container").is_in(
                        ["MED BAG", "MED BOX", "MED PKG", "MED PACK"]
                    )
                    & (pl.col("l_quantity").is_between(10, 20))
                    & (pl.col("p_size").is_between(1, 10))
                )
                | (
                    (pl.col("p_brand") == "Brand#34")
                    & pl.col("p_container").is_in(
                        ["LG CASE", "LG BOX", "LG PACK", "LG PKG"]
                    )
                    & (pl.col("l_quantity").is_between(20, 30))
                    & (pl.col("p_size").is_between(1, 15))
                )
            )
            .select(
                (pl.col("l_extendedprice") * (1 - pl.col("l_discount")))
                .sum()
                .round(2)
                .alias("revenue")
            )
        )

    @staticmethod
    def q20(run_config: RunConfig) -> pl.LazyFrame:
        """Query 20."""
        lineitem = get_data(run_config.dataset_path, "lineitem", run_config.suffix)
        nation = get_data(run_config.dataset_path, "nation", run_config.suffix)
        part = get_data(run_config.dataset_path, "part", run_config.suffix)
        partsupp = get_data(run_config.dataset_path, "partsupp", run_config.suffix)
        supplier = get_data(run_config.dataset_path, "supplier", run_config.suffix)

        var1 = date(1994, 1, 1)
        var2 = date(1995, 1, 1)
        var3 = "CANADA"
        var4 = "forest"

        q1 = (
            lineitem.filter(pl.col("l_shipdate").is_between(var1, var2, closed="left"))
            .group_by("l_partkey", "l_suppkey")
            .agg((pl.col("l_quantity").sum() * 0.5).alias("sum_quantity"))
        )
        q2 = nation.filter(pl.col("n_name") == var3)
        q3 = supplier.join(q2, left_on="s_nationkey", right_on="n_nationkey")

        return (
            part.filter(pl.col("p_name").str.starts_with(var4))
            .select(pl.col("p_partkey").unique())
            .join(partsupp, left_on="p_partkey", right_on="ps_partkey")
            .join(
                q1,
                left_on=["ps_suppkey", "p_partkey"],
                right_on=["l_suppkey", "l_partkey"],
            )
            .filter(pl.col("ps_availqty") > pl.col("sum_quantity"))
            .select(pl.col("ps_suppkey").unique())
            .join(q3, left_on="ps_suppkey", right_on="s_suppkey")
            .select("s_name", "s_address")
            .sort("s_name")
        )

    @staticmethod
    def q21(run_config: RunConfig) -> pl.LazyFrame:
        """Query 21."""
        lineitem = get_data(run_config.dataset_path, "lineitem", run_config.suffix)
        nation = get_data(run_config.dataset_path, "nation", run_config.suffix)
        orders = get_data(run_config.dataset_path, "orders", run_config.suffix)
        supplier = get_data(run_config.dataset_path, "supplier", run_config.suffix)

        var1 = "SAUDI ARABIA"

        q1 = (
            lineitem.group_by("l_orderkey")
            .agg(pl.col("l_suppkey").len().alias("n_supp_by_order"))
            .filter(pl.col("n_supp_by_order") > 1)
            .join(
                lineitem.filter(pl.col("l_receiptdate") > pl.col("l_commitdate")),
                on="l_orderkey",
            )
        )

        return (
            q1.group_by("l_orderkey")
            .agg(pl.col("l_suppkey").len().alias("n_supp_by_order"))
            .join(q1, on="l_orderkey")
            .join(supplier, left_on="l_suppkey", right_on="s_suppkey")
            .join(nation, left_on="s_nationkey", right_on="n_nationkey")
            .join(orders, left_on="l_orderkey", right_on="o_orderkey")
            .filter(pl.col("n_supp_by_order") == 1)
            .filter(pl.col("n_name") == var1)
            .filter(pl.col("o_orderstatus") == "F")
            .group_by("s_name")
            .agg(pl.len().alias("numwait"))
            .sort(by=["numwait", "s_name"], descending=[True, False])
            .head(100)
        )

    @staticmethod
    def q22(run_config: RunConfig) -> pl.LazyFrame:
        """Query 22."""
        customer = get_data(run_config.dataset_path, "customer", run_config.suffix)
        orders = get_data(run_config.dataset_path, "orders", run_config.suffix)

        q1 = (
            customer.with_columns(pl.col("c_phone").str.slice(0, 2).alias("cntrycode"))
            .filter(pl.col("cntrycode").str.contains("13|31|23|29|30|18|17"))
            .select("c_acctbal", "c_custkey", "cntrycode")
        )

        q2 = q1.filter(pl.col("c_acctbal") > 0.0).select(
            pl.col("c_acctbal").mean().alias("avg_acctbal")
        )

        q3 = orders.select(pl.col("o_custkey").unique()).with_columns(
            pl.col("o_custkey").alias("c_custkey")
        )

        return (
            q1.join(q3, on="c_custkey", how="left")
            .filter(pl.col("o_custkey").is_null())
            .join(q2, how="cross")
            .filter(pl.col("c_acctbal") > pl.col("avg_acctbal"))
            .group_by("cntrycode")
            .agg(
                pl.col("c_acctbal").count().alias("numcust"),
                pl.col("c_acctbal").sum().round(2).alias("totacctbal"),
            )
            .sort("cntrycode")
        )


if __name__ == "__main__":
    run_polars(PDSHQueries)
