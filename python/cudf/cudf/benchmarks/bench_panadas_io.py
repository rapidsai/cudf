# Copyright (c) 2020, NVIDIA CORPORATION.

import io
import glob
import os
import pandas as pd
import pytest

from bench_cudf_io import get_dataset_dir, get_dtypes
from cudf._fuzz_testing import utils
from get_datasets import create_dataset

datatype = ['float32', 'float64',
            'int32', 'int64',
            'str', 'datetime64[s]']

null_frequency = [0.1, 0.4, 0.8]


@pytest.mark.parametrize("dtype", datatype)
@pytest.mark.parametrize("null_frequency", null_frequency)
def bench_to_csv(benchmark,  dtype, null_frequency, run_bench):
    if not run_bench:
        pytest.skip("Pytest variable run_bench not passed as True")
    table, file_path = create_dataset(dtype, file_type="csv",
                                      only_file=False,
                                      null_frequency=null_frequency)

    pd_df = utils.pyarrow_to_pandas(table)
    benchmark(pd_df.to_csv, file_path)


@pytest.mark.parametrize("dtype", datatype)
def bench_from_csv(benchmark, use_buffer, dtype, run_bench):
    if not run_bench:
        pytest.skip("Pytest variable run_bench not passed as True")
    file_path = create_dataset(dtype, file_type="csv",
                               only_file=True,
                               null_frequency=None)

    if use_buffer == "True":
        with open(file_path, "rb") as f:
            file = io.BytesIO(f.read())
    else:
        file = file_path
    benchmark(pd.read_csv, file)
    os.remove(file_path)


@pytest.mark.parametrize("dtype", datatype)
def bench_read_orc(benchmark, use_buffer, dtype, run_bench):
    if not run_bench:
        pytest.skip("Pytest variable run_bench not passed as True")
    file_path = create_dataset(dtype, file_type="orc",
                               only_file=True,
                               null_frequency=None)

    if ~os.path.isfile(file_path):
        pytest.skip("no ORC file to read")

    if use_buffer == "True":
        with open(file_path, "rb") as f:
            file = io.BytesIO(f.read())
    else:
        file = file_path
    benchmark(pd.read_orc, file)
    os.remove(file_path)


@pytest.mark.parametrize("dtype", datatype)
@pytest.mark.parametrize("null_frequency", null_frequency)
def bench_to_parquet(benchmark,  dtype, null_frequency, run_bench):
    if not run_bench:
        pytest.skip("Pytest variable run_bench not passed as True")
    table, file_path = create_dataset(dtype, file_type="csv",
                                      only_file=False,
                                      null_frequency=null_frequency)

    pd_df = cudf._fuzz_testing.utils.pyarrow_to_pandas(table)
    benchmark(pd_df.to_parquet, file_path)


@pytest.mark.parametrize("dtype", datatype)
def bench_read_parquet(benchmark, use_buffer, dtype, run_bench):
    if not run_bench:
        pytest.skip("Pytest variable run_bench not passed as True")
    file_path = create_dataset(dtype, file_type="parquet",
                               only_file=True,
                               null_frequency=None)

    if use_buffer == "True":
        with open(file_path, "rb") as f:
            file = io.BytesIO(f.read())
    else:
        file = file_path
    benchmark(pd.read_parquet, file)
    os.remove(file_path)


@pytest.mark.parametrize("dtype", ["infer", "provide"])
@pytest.mark.parametrize("file_path", glob.glob(get_dataset_dir() + "json_*"))
def bench_json(benchmark, file_path, use_buffer, dtype):
    if "bz2" in file_path:
        compression = "bz2"
    elif "gzip" in file_path:
        compression = "gzip"
    elif "infer" in file_path:
        compression = "infer"
    else:
        raise TypeError("Unsupported compression type")

    if dtype == "infer":
        dtype = True
    else:
        dtype = get_dtypes(file_path)

    if use_buffer == "True":
        with open(file_path, "rb") as f:
            file_path = io.BytesIO(f.read())

    benchmark(
        pd.read_json,
        file_path,
        compression=compression,
        lines=True,
        orient="records",
        dtype=dtype,
    )
