# Copyright (c) 2020-2021, NVIDIA CORPORATION.

import io
import glob
import os
import pandas as pd
import pytest

from bench_cudf_io import get_dtypes
from conftest import option
from get_datasets import create_dataset

datatype = ["float32", "float64", "int32", "int64", "str", "datetime64[s]"]

null_frequency = [0.1, 0.4, 0.8]


@pytest.mark.skipif(
    option.bench_pandas is False,
    reason="Pass `bench_pandas` as True to run panda benchmarks",
)
@pytest.mark.parametrize("dtype", datatype)
@pytest.mark.parametrize("null_frequency", null_frequency)
def bench_to_csv(benchmark, dtype, null_frequency, run_bench):
    from cudf._fuzz_testing import utils
    table, file_path = create_dataset(
        dtype, file_type="csv", only_file=False, null_frequency=null_frequency
    )

    pd_df = utils.pyarrow_to_pandas(table)
    benchmark(pd_df.to_csv, file_path)


@pytest.mark.skipif(
    option.bench_pandas is False,
    reason="Pass `bench_pandas` as True to run panda benchmarks",
)
@pytest.mark.parametrize("dtype", datatype)
def bench_from_csv(benchmark, use_buffer, dtype, run_bench):
    file_path = create_dataset(
        dtype, file_type="csv", only_file=True, null_frequency=None
    )

    if use_buffer == "True":
        with open(file_path, "rb") as f:
            file = io.BytesIO(f.read())
    else:
        file = file_path
    benchmark(pd.read_csv, file)
    os.remove(file_path)


@pytest.mark.skipif(
    option.bench_pandas is False,
    reason="Pass `bench_pandas` as True to run panda benchmarks",
)
@pytest.mark.parametrize("dtype", datatype)
def bench_read_orc(benchmark, use_buffer, dtype, run_bench):
    file_path = create_dataset(
        dtype, file_type="orc", only_file=True, null_frequency=None
    )

    if ~os.path.isfile(file_path):
        pytest.skip("no ORC file to read")

    if use_buffer == "True":
        with open(file_path, "rb") as f:
            file = io.BytesIO(f.read())
    else:
        file = file_path
    benchmark(pd.read_orc, file)
    os.remove(file_path)


@pytest.mark.skipif(
    option.bench_pandas is False,
    reason="Pass `bench_pandas` as True to run panda benchmarks",
)
@pytest.mark.parametrize("dtype", datatype)
@pytest.mark.parametrize("null_frequency", null_frequency)
def bench_to_parquet(benchmark, dtype, null_frequency, run_bench):
    from cudf._fuzz_testing import utils
    table, file_path = create_dataset(
        dtype, file_type="csv", only_file=False, null_frequency=null_frequency
    )

    pd_df = utils.pyarrow_to_pandas(table)
    benchmark(pd_df.to_parquet, file_path)


@pytest.mark.skipif(
    option.bench_pandas is False,
    reason="Pass `bench_pandas` as True to run panda benchmarks",
)
@pytest.mark.parametrize("dtype", datatype)
def bench_read_parquet(benchmark, use_buffer, dtype, run_bench):
    file_path = create_dataset(
        dtype, file_type="parquet", only_file=True, null_frequency=None
    )

    if use_buffer == "True":
        with open(file_path, "rb") as f:
            file = io.BytesIO(f.read())
    else:
        file = file_path
    benchmark(pd.read_parquet, file)
    os.remove(file_path)


@pytest.mark.skipif(
    option.bench_pandas is False,
    reason="Pass `bench_pandas` as True to run panda benchmarks",
)
@pytest.mark.parametrize("dtype", ["infer", "provide"])
def bench_json(benchmark, file_path, use_buffer, dtype, dataset_dir):
    file_path = glob.glob(dataset_dir + "json_*")
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
