# Copyright (c) 2020, NVIDIA CORPORATION.

import glob
import io
import os
import pandas as pd
import pytest

from bench_cudf_io import get_dataset_dir, get_dtypes
from get_datasets import create_pandas_dataset


@pytest.mark.parametrize("dtype", ['float32', 'float64',
                                   'int32', 'int64',
                                   'str', 'datetime64[s]'])
def bench_to_csv(benchmark, dtype):
    pd_df, file_path = create_pandas_dataset(dtype, file_type="csv",
                                             only_file=False)
    benchmark(pd_df.to_csv, file_path)


@pytest.mark.parametrize("dtype", ['float32', 'float64',
                                   'int32', 'int64',
                                   'str', 'datetime64[s]'])
def bench_from_csv(benchmark, use_buffer, dtype):
    file_path = create_pandas_dataset(dtype, file_type="csv",
                                      only_file=True)

    if use_buffer == "True":
        with open(file_path, "rb") as f:
            file = io.BytesIO(f.read())
    else:
        file = file_path
    benchmark(pd.read_csv, file)
    os.remove(file_path)


@pytest.mark.parametrize("dtype", ['float32', 'float64',
                                   'int32', 'int64',
                                   'str', 'datetime64[s]'])
def bench_read_orc(benchmark, use_buffer, dtype):
    file_path = create_pandas_dataset(dtype, file_type="orc",
                                      only_file=True)
    if use_buffer == "True":
        with open(file_path, "rb") as f:
            file = io.BytesIO(f.read())
    else:
        file = file_path
    benchmark(pd.read_orc, file)
    os.remove(file_path)


@pytest.mark.parametrize("dtype", ['float32', 'float64',
                                   'int32', 'int64',
                                   'str', 'datetime64[s]'])
def bench_to_parquet(benchmark, dtype):
    pd_df, file_path = create_pandas_dataset(dtype, file_type="parquet",
                                             only_file=False)
    benchmark(pd_df.to_parquet, file_path)


@pytest.mark.parametrize("dtype", ['float32', 'float64',
                                   'int32', 'int64',
                                   'str', 'datetime64[s]'])
def bench_read_parquet(benchmark, use_buffer, dtype):
    file_path = create_pandas_dataset(dtype, file_type="parquet",
                                      only_file=True)
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
