# Copyright (c) 2019-2022, NVIDIA CORPORATION.

from functools import partial

import dask

import cudf

read_json = partial(dask.dataframe.read_json, engine=cudf.read_json)
