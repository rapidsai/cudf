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

from datetime import date
from typing import TYPE_CHECKING

import pandas as pd

from cudf.pandas._benchmarks.utils import (
    get_data,
    run_pandas,
)

if TYPE_CHECKING:
    from cudf.pandas._benchmarks.utils import RunConfig


class PDSHQueries:
    """PDS-H query definitions."""

    name: str = "pdsh"

    @staticmethod
    def q0(run_config: RunConfig) -> pd.DataFrame:
        """Query 0."""
        return pd.DataFrame()

    @staticmethod
    def q1(run_config: RunConfig) -> pd.DataFrame:
        """Query 1."""
        line_item_ds = get_data(
            run_config.dataset_path, "lineitem", run_config.suffix
        )

        var1 = date(1998, 9, 2)

        filt = line_item_ds[line_item_ds["l_shipdate"] <= var1]

        # This is lenient towards pandas as normally an optimizer should decide
        # that this could be computed before the groupby aggregation.
        # Other implementations don't enjoy this benefit.
        filt["disc_price"] = filt.l_extendedprice * (1.0 - filt.l_discount)
        filt["charge"] = (
            filt.l_extendedprice * (1.0 - filt.l_discount) * (1.0 + filt.l_tax)
        )

        gb = filt.groupby(["l_returnflag", "l_linestatus"], as_index=False)
        agg = gb.agg(
            sum_qty=pd.NamedAgg(column="l_quantity", aggfunc="sum"),
            sum_base_price=pd.NamedAgg(
                column="l_extendedprice", aggfunc="sum"
            ),
            sum_disc_price=pd.NamedAgg(column="disc_price", aggfunc="sum"),
            sum_charge=pd.NamedAgg(column="charge", aggfunc="sum"),
            avg_qty=pd.NamedAgg(column="l_quantity", aggfunc="mean"),
            avg_price=pd.NamedAgg(column="l_extendedprice", aggfunc="mean"),
            avg_disc=pd.NamedAgg(column="l_discount", aggfunc="mean"),
            count_order=pd.NamedAgg(column="l_orderkey", aggfunc="size"),
        )

        return agg.sort_values(["l_returnflag", "l_linestatus"])

    @staticmethod
    def q2(run_config: RunConfig) -> pd.DataFrame:
        """Query 2."""
        nation = get_data(run_config.dataset_path, "nation", run_config.suffix)
        part = get_data(run_config.dataset_path, "part", run_config.suffix)
        partsupp = get_data(
            run_config.dataset_path, "partsupp", run_config.suffix
        )
        region = get_data(run_config.dataset_path, "region", run_config.suffix)
        supplier = get_data(
            run_config.dataset_path, "supplier", run_config.suffix
        )

        var1 = 15
        var2 = "BRASS"
        var3 = "EUROPE"

        jn = (
            part.merge(partsupp, left_on="p_partkey", right_on="ps_partkey")
            .merge(supplier, left_on="ps_suppkey", right_on="s_suppkey")
            .merge(nation, left_on="s_nationkey", right_on="n_nationkey")
            .merge(region, left_on="n_regionkey", right_on="r_regionkey")
        )

        jn = jn[jn["p_size"] == var1]
        jn = jn[jn["p_type"].str.endswith(var2)]
        jn = jn[jn["r_name"] == var3]

        gb = jn.groupby("p_partkey", as_index=False)
        agg = gb["ps_supplycost"].min()
        jn2 = agg.merge(jn, on=["p_partkey", "ps_supplycost"])

        sel = jn2.loc[
            :,
            [
                "s_acctbal",
                "s_name",
                "n_name",
                "p_partkey",
                "p_mfgr",
                "s_address",
                "s_phone",
                "s_comment",
            ],
        ]

        sort = sel.sort_values(
            by=["s_acctbal", "n_name", "s_name", "p_partkey"],
            ascending=[False, True, True, True],
        )
        return sort.head(100)

    @staticmethod
    def q3(run_config: RunConfig) -> pd.DataFrame:
        """Query 3."""
        customer = get_data(
            run_config.dataset_path, "customer", run_config.suffix
        )
        lineitem = get_data(
            run_config.dataset_path, "lineitem", run_config.suffix
        )
        orders = get_data(run_config.dataset_path, "orders", run_config.suffix)

        var1 = "BUILDING"
        var2 = date(1995, 3, 15)

        fcustomer = customer[customer["c_mktsegment"] == var1]

        jn1 = fcustomer.merge(
            orders, left_on="c_custkey", right_on="o_custkey"
        )
        jn2 = jn1.merge(lineitem, left_on="o_orderkey", right_on="l_orderkey")

        jn2 = jn2[jn2["o_orderdate"] < var2]
        jn2 = jn2[jn2["l_shipdate"] > var2]
        jn2["revenue"] = jn2.l_extendedprice * (1 - jn2.l_discount)

        gb = jn2.groupby(
            ["o_orderkey", "o_orderdate", "o_shippriority"], as_index=False
        )
        agg = gb["revenue"].sum()

        sel = agg.loc[
            :, ["o_orderkey", "revenue", "o_orderdate", "o_shippriority"]
        ]
        sel = sel.rename(columns={"o_orderkey": "l_orderkey"})

        sorted_df = sel.sort_values(
            by=["revenue", "o_orderdate"], ascending=[False, True]
        )
        return sorted_df.head(10)

    @staticmethod
    def q4(run_config: RunConfig) -> pd.DataFrame:
        """Query 4."""
        lineitem = get_data(
            run_config.dataset_path, "lineitem", run_config.suffix
        )
        orders = get_data(run_config.dataset_path, "orders", run_config.suffix)

        var1 = date(1993, 7, 1)
        var2 = date(1993, 10, 1)

        jn = lineitem.merge(
            orders, left_on="l_orderkey", right_on="o_orderkey"
        )

        jn = jn[(jn["o_orderdate"] >= var1) & (jn["o_orderdate"] < var2)]
        jn = jn[jn["l_commitdate"] < jn["l_receiptdate"]]

        jn = jn.drop_duplicates(subset=["o_orderpriority", "l_orderkey"])

        gb = jn.groupby("o_orderpriority", as_index=False)
        agg = gb.agg(
            order_count=pd.NamedAgg(column="o_orderkey", aggfunc="count")
        )

        return agg.sort_values(["o_orderpriority"])

    @staticmethod
    def q5(run_config: RunConfig) -> pd.DataFrame:
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

        jn1 = region.merge(
            nation, left_on="r_regionkey", right_on="n_regionkey"
        )
        jn2 = jn1.merge(
            customer, left_on="n_nationkey", right_on="c_nationkey"
        )
        jn3 = jn2.merge(orders, left_on="c_custkey", right_on="o_custkey")
        jn4 = jn3.merge(lineitem, left_on="o_orderkey", right_on="l_orderkey")
        jn5 = jn4.merge(
            supplier,
            left_on=["l_suppkey", "n_nationkey"],
            right_on=["s_suppkey", "s_nationkey"],
        )

        jn5 = jn5[jn5["r_name"] == var1]
        jn5 = jn5[(jn5["o_orderdate"] >= var2) & (jn5["o_orderdate"] < var3)]
        jn5["revenue"] = jn5.l_extendedprice * (1.0 - jn5.l_discount)

        gb = jn5.groupby("n_name", as_index=False)["revenue"].sum()
        return gb.sort_values("revenue", ascending=False)

    @staticmethod
    def q6(run_config: RunConfig) -> pd.DataFrame:
        """Query 6."""
        path = run_config.dataset_path
        suffix = run_config.suffix
        lineitem = get_data(path, "lineitem", suffix)

        var1 = date(1994, 1, 1)
        var2 = date(1995, 1, 1)
        var3 = 0.05
        var4 = 0.07
        var5 = 24

        filt = lineitem[
            (lineitem["l_shipdate"] >= var1) & (lineitem["l_shipdate"] < var2)
        ]
        filt = filt[
            (filt["l_discount"] >= var3) & (filt["l_discount"] <= var4)
        ]
        filt = filt[filt["l_quantity"] < var5]
        result_value = (filt["l_extendedprice"] * filt["l_discount"]).sum()
        return pd.DataFrame({"revenue": [result_value]})


if __name__ == "__main__":
    run_pandas(PDSHQueries)
