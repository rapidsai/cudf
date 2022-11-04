# Copyright (c) 2019-2022, NVIDIA CORPORATION.

from functools import partial

import dask.dataframe as dd

import cudf

read_json = partial(dd.read_json, engine=cudf.read_json)
read_json_experimental = partial(
    dd.read_json, engine=partial(cudf.read_json, engine="cudf_experimental")
)
