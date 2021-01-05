import cudf
import io
import os
import pytest
import pandas as pd


path = "gcs://anaconda-public-data/nyc-taxi/csv/"
# NYC taxi data from January 2016
cudf_df = cudf.read_csv(path + "2016/yellow_tripdata_2016-01.csv")


def obtain_file_path(file_type):
    file_dir = "cudf/benchmarks/cuio_data/"
    file_path = os.path.join(file_dir, "file_data" + "." + file_type)
    return file_path


def bench_to_json(benchmark):
    file_path = obtain_file_path(file_type="json")
    benchmark(cudf_df.to_json, file_path)


def bench_read_json(benchmark, use_buffer):
    file_path = obtain_file_path(file_type="json")
    if use_buffer == "True":
        with open(file_path, "rb") as f:
            file = io.BytesIO(f.read())
    else:
        file = file_path
    benchmark(cudf.read_json, file)
    os.remove(file_path)


def bench_to_csv(benchmark):
    file_path = obtain_file_path(file_type="csv")
    benchmark(cudf_df.to_csv, file_path)


def bench_read_csv(benchmark, use_buffer):
    file_path = obtain_file_path(file_type="csv")
    if use_buffer == "True":
        with open(file_path, "rb") as f:
            file = io.BytesIO(f.read())
    else:
        file = file_path
    benchmark(cudf.read_csv, file)
    os.remove(file_path)


def bench_to_orc(benchmark):
    file_path = obtain_file_path(file_type="orc")
    benchmark(cudf_df.to_orc, file_path)


def bench_read_orc(benchmark, use_buffer, bench_pandas):
    file_path = obtain_file_path(file_type="orc")
    if use_buffer == "True":
        with open(file_path, "rb") as f:
            file = io.BytesIO(f.read())
    else:
        file = file_path
    benchmark(cudf.read_orc, file)
    if not bench_pandas:
        os.remove(file_path)


def bench_to_parquet(benchmark):
    file_path = obtain_file_path(file_type="parquet")
    benchmark(cudf_df.to_parquet, file_path)


def bench_read_parquet(benchmark, use_buffer):
    file_path = obtain_file_path(file_type="parquet")
    if use_buffer == "True":
        with open(file_path, "rb") as f:
            file = io.BytesIO(f.read())
    else:
        file = file_path
    benchmark(cudf.read_parquet, file)
    os.remove(file_path)


# Run panda benchmarks if bench_pandas=True
def bench_pandas_to_csv(benchmark, bench_pandas):
    if not bench_pandas:
        pytest.skip(
            "bench_pandas=False, panda functions " " will not be benchmarked"
        )
    file_path = obtain_file_path(file_type="csv")
    print("file_path : ", file_path)
    pd_df = cudf_df.to_pandas()
    benchmark(pd_df.to_csv, file_path)


def bench_pandas_read_csv(benchmark, use_buffer, bench_pandas):
    if not bench_pandas:
        pytest.skip(
            "bench_pandas=False, panda functions " " will not be benchmarked"
        )
    file_path = obtain_file_path(file_type="csv")
    print("file_path : ", file_path)

    if use_buffer == "True":
        with open(file_path, "rb") as f:
            file = io.BytesIO(f.read())
    else:
        file = file_path
    benchmark(pd.read_csv, file)
    os.remove(file_path)


def bench_pandas_read_orc(benchmark, use_buffer, bench_pandas):
    if not bench_pandas:
        pytest.skip(
            "bench_pandas=False, panda functions " " will not be benchmarked"
        )
    file_path = obtain_file_path(file_type="orc")
    if use_buffer == "True":
        with open(file_path, "rb") as f:
            file = io.BytesIO(f.read())
    else:
        file = file_path
    benchmark(pd.read_orc, file)
    os.remove(file_path)


def bench_pandas_to_parquet(benchmark, bench_pandas):
    if not bench_pandas:
        pytest.skip(
            "bench_pandas=False, panda functions " " will not be benchmarked"
        )
    file_path = obtain_file_path(file_type="parquet")
    pd_df = cudf_df.to_pandas()
    benchmark(pd_df.to_parquet, file_path)


def bench_pandas_read_parquet(benchmark, use_buffer, bench_pandas):
    if not bench_pandas:
        pytest.skip(
            "bench_pandas=False, panda functions " " will not be benchmarked"
        )
    file_path = obtain_file_path(file_type="parquet")
    if use_buffer == "True":
        with open(file_path, "rb") as f:
            file = io.BytesIO(f.read())
    else:
        file = file_path
    benchmark(pd.read_parquet, file)
    os.remove(file_path)


def bench_pandas_to_json(benchmark, bench_pandas):
    if not bench_pandas:
        pytest.skip(
            "bench_pandas=False, panda functions " " will not be benchmarked"
        )
    file_path = obtain_file_path(file_type="json")
    pd_df = cudf_df.to_pandas()
    benchmark(pd_df.to_json, file_path)


def bench_pandas_read_json(benchmark, use_buffer):
    file_path = obtain_file_path(file_type="json")
    if use_buffer == "True":
        with open(file_path, "rb") as f:
            file = io.BytesIO(f.read())
    else:
        file = file_path
    benchmark(pd.read_json, file)
    os.remove(file_path)
