# Copyright (c) 2022, NVIDIA CORPORATION.

import argparse
import os
import time

import dask
import dask.dataframe
import dask_cuda
from distributed import Client, wait

import cudf

NROWS = 100_000_000


def main(args):
    os.environ["CUDF_SPILL"] = "on" if args.spill == "cudf" else "off"

    device_memory_limit = None
    if args.spill == "dask":
        device_memory_limit = 0.5
    elif args.spill == "jit":
        device_memory_limit = 0.8

    cluster = dask_cuda.LocalCUDACluster(
        protocol="tcp",
        n_workers=args.n_workers,
        device_memory_limit=device_memory_limit,
        memory_limit=None,
        jit_unspill=args.spill == "jit",
    )

    with Client(cluster):
        t1 = time.monotonic()
        meta = cudf.datasets.randomdata(nrows=1)
        df: dask.dataframe.DataFrame = dask.dataframe.from_map(
            lambda x: cudf.datasets.randomdata(nrows=NROWS),
            range(args.npartitions),
            meta=meta,
        )
        df = df.persist()
        wait(df)
        res = df.shuffle(on="x")
        print(res)
        res = res.persist()
        wait(res)
        t2 = time.monotonic()
        print(res.head())
        print("time: ", t2 - t1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--spill",
        choices=["dask", "jit", "cudf"],
        default="cudf",
        type=str,
        help="The spilling backend",
    )
    parser.add_argument(
        "--npartitions",
        default=10,
        type=int,
    )
    parser.add_argument(
        "-n",
        "--n-workers",
        default=1,
        type=int,
    )

    main(parser.parse_args())
