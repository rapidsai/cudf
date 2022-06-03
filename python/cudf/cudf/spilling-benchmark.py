# Copyright (c) 2022, NVIDIA CORPORATION.

import argparse
import os
import time

import dask
import dask.dataframe
from distributed import Client, LocalCluster, wait

import cudf

NROWS = 130_000_000


def main(args):
    os.environ["CUDF_SPILL"] = "on" if args.spill == "cudf" else "off"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    device_memory_limit = None
    if args.spill == "dask":
        device_memory_limit = 0.5
    elif args.spill == "jit":
        device_memory_limit = 0.8

    if args.spill == "cudf":
        cluster = LocalCluster(
            protocol="tcp",
            n_workers=1,
            threads_per_worker=1,
            memory_limit=None,
        )
    else:
        import dask_cuda

        cluster = dask_cuda.LocalCUDACluster(
            protocol="tcp",
            n_workers=1,
            device_memory_limit=device_memory_limit,
            memory_limit="auto",
            jit_unspill=args.spill == "jit",
            CUDA_VISIBLE_DEVICES=[1],
        )

    with Client(cluster):
        t1 = time.monotonic()
        meta = cudf.datasets.randomdata(nrows=1)
        df: dask.dataframe.DataFrame = dask.dataframe.from_map(
            lambda x: cudf.datasets.randomdata(nrows=NROWS),
            range(args.npartitions),
            meta=meta,
        )
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
        choices=["none", "dask", "jit", "cudf"],
        default="none",
        type=str,
        help="The spilling backend",
    )
    parser.add_argument(
        "--npartitions",
        default=10,
        type=int,
    )

    main(parser.parse_args())
