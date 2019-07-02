from functools import partial

import cudf
import dask

read_json = partial(dask.dataframe.read_json, engine=cudf.read_json)
