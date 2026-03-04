# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from dask_cudf.core import _deprecated_api

from . import csv, json, orc, parquet, text  # noqa: F401

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
    new_api="dask_cudf.io.orc.to_orc",
    rec="Please use the DataFrame.to_orc method instead.",
)
read_text = _deprecated_api(
    "dask_cudf.io.read_text", new_api="dask_cudf.read_text"
)
read_parquet = parquet.read_parquet
to_parquet = _deprecated_api(
    "dask_cudf.io.to_parquet",
    new_api="dask_cudf._legacy.io.parquet.to_parquet",
    rec="Please use the DataFrame.to_parquet method instead.",
)
