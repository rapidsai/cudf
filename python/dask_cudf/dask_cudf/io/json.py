from functools import partial

import dask

import cudf

read_json = partial(dask.dataframe.read_json, engine=cudf.read_json)
