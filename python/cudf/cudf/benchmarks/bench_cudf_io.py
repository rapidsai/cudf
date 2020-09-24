# Copyright (c) 2018-2020, NVIDIA CORPORATION.

import pytest
import cudf
import glob
import io


@pytest.mark.parametrize(
    "file_path", glob.glob("cudf/benchmarks/cuio_data/datasets/avro_*")
)
def bench_avro(benchmark, file_path, use_buffer):

    if use_buffer == "True":
        with open(file_path, "rb") as f:
            file_path = io.BytesIO(f.read())
    benchmark(cudf.read_avro, file_path)


@pytest.mark.parametrize(
    "file_path", glob.glob("cudf/benchmarks/cuio_data/datasets/json_*")
)
def bench_json(benchmark, file_path, use_buffer):
    if "bz2" in file_path:
        compression = "bz2"
    elif "gzip" in file_path:
        compression = "gzip"
    elif "infer" in file_path:
        compression = "infer"
    else:
        raise TypeError("Unsupported compression type")

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
    )
