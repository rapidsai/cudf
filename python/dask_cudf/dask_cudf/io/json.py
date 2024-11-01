# Copyright (c) 2024, NVIDIA CORPORATION.

from dask_cudf import _deprecated_api

read_json = _deprecated_api(
    "dask_cudf.io.json.read_json",
    new_api="dask_cudf.read_json",
)
