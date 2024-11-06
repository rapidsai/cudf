# Copyright (c) 2024, NVIDIA CORPORATION.

from dask_cudf import _deprecated_api

read_csv = _deprecated_api(
    "dask_cudf.io.csv.read_csv",
    new_api="dask_cudf.read_csv",
)
