# Copyright (c) 2020, NVIDIA CORPORATION.

import glob
import io

import pytest
from conftest import option

import cudf


def get_dataset_dir():
    if option.dataset_dir == "NONE":
        return "cudf/benchmarks/cuio_data/datasets/"
    return option.dataset_dir


@pytest.mark.parametrize("skiprows", [None, 100000, 200000])
@pytest.mark.parametrize("file_path", glob.glob(get_dataset_dir() + "avro_*"))
def bench_avro(benchmark, file_path, use_buffer, skiprows):

    if use_buffer == "True":
        with open(file_path, "rb") as f:
            file_path = io.BytesIO(f.read())
    benchmark(cudf.read_avro, file_path, skiprows=skiprows)


def get_dtypes(file_path):
    if "_unsigned_int_" in file_path:
        return ["uint8", "uint16", "uint32", "uint64"] * 16
    elif "_int_" in file_path:
        return ["int8", "int16", "int32", "int64"] * 16
    elif "_float_" in file_path:
        return ["float32", "float64"] * 32
    elif "_str_" in file_path:
        return ["str"] * 64
    elif "_datetime64_" in file_path:
        return [
            "timestamp[s]",
            "timestamp[ms]",
            "timestamp[us]",
            "timestamp[ns]",
        ] * 16
    elif "_timedelta64_" in file_path:
        return [
            "timedelta64[s]",
            "timedelta64[ms]",
            "timedelta64[us]",
            "timedelta64[ns]",
        ] * 16
    elif "_bool_" in file_path:
        return ["bool"] * 64
    else:
        raise TypeError("Unsupported dtype file")


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
        cudf.read_json,
        file_path,
        engine="cudf",
        compression=compression,
        lines=True,
        orient="records",
        dtype=dtype,
    )
