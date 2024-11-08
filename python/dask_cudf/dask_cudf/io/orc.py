# Copyright (c) 2024, NVIDIA CORPORATION.

from dask_cudf import _deprecated_api

read_orc = _deprecated_api(
    "dask_cudf.io.orc.read_orc",
    new_api="dask_cudf.read_orc",
)
to_orc = _deprecated_api(
    "dask_cudf.io.orc.to_orc",
    new_api="dask_cudf._legacy.io.orc.to_orc",
    rec="Please use the DataFrame.to_orc method instead.",
)
