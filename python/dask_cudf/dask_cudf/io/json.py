import cudf
import dask
from functools import partial


read_json = partial(dask.dataframe.read_json, engine=cudf.read_json)
