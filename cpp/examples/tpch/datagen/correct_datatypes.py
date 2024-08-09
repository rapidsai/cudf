# Copyright (c) 2024, NVIDIA CORPORATION.

import os
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd

if __name__ == "__main__":
    dataset_path = str(sys.argv[1])
    tables = ["lineitem", "part", "partsupp", "orders", "supplier", "customer", "nation", "region"]
    for table in tables:
        filepath = os.path.join(dataset_path, f"{table}.parquet")
        print("Reading file ", filepath)

        if filepath.endswith("lineitem.parquet"):
            df = pd.read_parquet(filepath)
            df["l_linenumber"] = df["l_linenumber"].astype("int64")
            df["l_quantity"] = df["l_quantity"].astype("int64")
            df["l_extendedprice"] = df["l_extendedprice"].astype("float64")
            df["l_discount"] = df["l_discount"].astype("float64")
            df["l_tax"] = df["l_tax"].astype("float64")
            pq.write_table(pa.Table.from_pandas(df), filepath, compression="snappy")

        elif filepath.endswith("part.parquet"):
            df = pd.read_parquet(filepath)
            df["p_size"] = df["p_size"].astype("int64")
            df["p_retailprice"] = df["p_retailprice"].astype("float64")
            pq.write_table(pa.Table.from_pandas(df), filepath, compression="snappy")

        elif filepath.endswith("partsupp.parquet"):
            df = pd.read_parquet(filepath)
            df["ps_availqty"] = df["ps_availqty"].astype("int64")
            df["ps_supplycost"] = df["ps_supplycost"].astype("float64")
            pq.write_table(pa.Table.from_pandas(df), filepath, compression="snappy")

        elif filepath.endswith("orders.parquet"):
            df = pd.read_parquet(filepath)
            df["o_totalprice"] = df["o_totalprice"].astype("float64")
            df["o_shippriority"] = df["o_shippriority"].astype("int64")
            pq.write_table(pa.Table.from_pandas(df), filepath, compression="snappy")

        elif filepath.endswith("supplier.parquet"):
            df = pd.read_parquet(filepath)
            df["s_acctbal"] = df["s_acctbal"].astype("float64")
            pq.write_table(pa.Table.from_pandas(df), filepath, compression="snappy")

        elif filepath.endswith("customer.parquet"):
            df = pd.read_parquet(filepath)
            df["c_acctbal"] = df["c_acctbal"].astype("float64")
            pq.write_table(pa.Table.from_pandas(df), filepath, compression="snappy")

        elif filepath.endswith("nation.parquet"):
            df = pd.read_parquet(filepath)
            pq.write_table(pa.Table.from_pandas(df), filepath, compression="snappy")

        elif filepath.endswith("region.parquet"):
            df = pd.read_parquet(filepath)
            pq.write_table(pa.Table.from_pandas(df), filepath, compression="snappy")
