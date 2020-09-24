# Copyright (c) 2018-2020, NVIDIA CORPORATION.

import pytest
import cudf
import glob

@pytest.mark.parametrize(file_path, glob.glob("cudf/benchmarks/datasets/datasets/avro_*"))
def bench_avro(benchmark, ):
    benchmark(cudf.read_avro, file_path)


@pytest.mark.parametrize(file_path, glob.glob("cudf/benchmarks/datasets/datasets/json_*"))
def bench_json(benchmark, file_path):
    if "bz2" in file_path:
        compression="bz2"
    elif "gzip" in file_path:
        compression="gzip"
    elif "infer" in file_path:
        compression="infer"
    else:
        raise TypeError("Unsupported compression type")
    benchmark(
        cudf.read_json,
        fname,
        engine="cudf",
        compression=compression,
        lines=True,
        orient="records",
    )
