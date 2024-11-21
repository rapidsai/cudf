# Copyright (c) 2024, NVIDIA CORPORATION.

from dask_cudf import _deprecated_api, QUERY_PLANNING_ON

from . import csv, orc, json, parquet, text  # noqa: F401


read_csv = _deprecated_api(
    "dask_cudf.io.read_csv", new_api="dask_cudf.read_csv"
)
read_json = _deprecated_api(
    "dask_cudf.io.read_json", new_api="dask_cudf.read_json"
)
read_orc = _deprecated_api(
    "dask_cudf.io.read_orc", new_api="dask_cudf.read_orc"
)
to_orc = _deprecated_api(
    "dask_cudf.io.to_orc",
    new_api="dask_cudf._legacy.io.to_orc",
    rec="Please use the DataFrame.to_orc method instead.",
)
read_text = _deprecated_api(
    "dask_cudf.io.read_text", new_api="dask_cudf.read_text"
)
if QUERY_PLANNING_ON:
    read_parquet = parquet.read_parquet
else:
    read_parquet = _deprecated_api(
        "The legacy dask_cudf.io.read_parquet API",
        new_api="dask_cudf.read_parquet",
        rec="",
    )
to_parquet = _deprecated_api(
    "dask_cudf.io.to_parquet",
    new_api="dask_cudf._legacy.io.parquet.to_parquet",
    rec="Please use the DataFrame.to_parquet method instead.",
)
