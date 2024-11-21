# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Experimental TPC-H benchmarks."""

from __future__ import annotations

import argparse
import time
from datetime import date

import polars as pl

from cudf_polars.dsl.translate import Translator
from cudf_polars.experimental.parallel import evaluate_dask

parser = argparse.ArgumentParser(
    prog="Cudf-Polars TPC-H Benchmarks",
    description="Experimental Dask-Executor benchmarks.",
)
parser.add_argument(
    "query",
    type=int,
    choices=[1, 5, 10, 18],
    help="Query number.",
)
parser.add_argument(
    "--path",
    type=str,
    default="/datasets/tpch_sf100",
    help="Root directory path.",
)
parser.add_argument(
    "--suffix",
    type=str,
    default="",
    help="Table file suffix.",
)
parser.add_argument(
    "-e",
    "--executor",
    default="dask-experimental",
    type=str,
    choices=["dask-experimental", "cudf", "polars"],
    help="Executor.",
)
parser.add_argument(
    "--blocksize",
    default=1 * 1024**3,
    type=int,
    help="Approx. partition size.",
)
parser.add_argument(
    "--debug",
    default=False,
    action="store_true",
    help="Debug run.",
)
args = parser.parse_args()


def get_data(path, table_name, suffix=""):
    """Get table from dataset."""
    return pl.scan_parquet(f"{path}/{table_name}{suffix}")


def q1(args):
    """Query 1."""
    lineitem = get_data(args.path, "lineitem", args.suffix)

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


def q5(args):
    """Query 5."""
    path = args.path
    suffix = args.suffix
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
            (pl.col("l_extendedprice") * (1 - pl.col("l_discount"))).alias("revenue")
        )
        .group_by("n_name")
        .agg(pl.sum("revenue"))
        .sort(by="revenue", descending=True)
    )


def q10(args):
    """Query 10."""
    path = args.path
    suffix = args.suffix
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
            # .round(2)  # TODO: Support `round`
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


def q18(args):
    """Query 18."""
    path = args.path
    suffix = args.suffix
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
        .group_by("c_name", "o_custkey", "o_orderkey", "o_orderdate", "o_totalprice")
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


def run(args):
    """Run the benchmark once."""
    t0 = time.time()

    q_id = args.query
    if q_id == 1:
        q = q1(args)
    elif q_id == 5:
        q = q5(args)
    elif q_id == 10:
        q = q10(args)
    elif q_id == 18:
        q = q18(args)
    else:
        raise NotImplementedError(f"Query {q_id} not implemented.")

    executor = args.executor
    if executor == "polars":
        result = q.collect()
    else:
        engine = pl.GPUEngine(
            raise_on_fail=True,
            executor=executor,
            parquet_options={"blocksize": args.blocksize, "chunked": False},
        )
        if args.debug:
            ir = Translator(q._ldf.visit(), engine).translate_ir()
            if args.executor == "cudf":
                result = ir.evaluate(cache={}).to_polars()
            elif args.executor == "dask-experimental":
                result = evaluate_dask(ir).to_polars()
        else:
            result = q.collect(engine=engine)

    t1 = time.time()
    print(result)
    print(f"time is {t1-t0}")


if __name__ == "__main__":
    run(args)
