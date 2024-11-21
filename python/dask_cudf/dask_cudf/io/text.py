# Copyright (c) 2024, NVIDIA CORPORATION.

from dask_cudf import _deprecated_api

read_text = _deprecated_api(
    "dask_cudf.io.text.read_text",
    new_api="dask_cudf.read_text",
)
