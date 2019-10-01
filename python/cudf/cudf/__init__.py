# Copyright (c) 2018-2019, NVIDIA CORPORATION.


from cudf import core, datasets
from cudf._version import get_versions
from cudf.core import DataFrame, Index, MultiIndex, Series, from_pandas, merge
from cudf.core.ops import (
    arccos,
    arcsin,
    arctan,
    cos,
    exp,
    log,
    logical_and,
    logical_not,
    logical_or,
    sin,
    sqrt,
    tan,
)
from cudf.core.reshape import concat, get_dummies, melt
from cudf.io import (
    from_dlpack,
    read_avro,
    read_csv,
    read_feather,
    read_hdf,
    read_json,
    read_orc,
    read_parquet,
)
from cudf.utils.utils import set_allocator

__version__ = get_versions()["version"]
del get_versions


# Import dask_cudf dispatch functions
try:
    from dask_cudf.backends import (
        hash_df_cudf,
        hash_df_cudf_index,
        group_split_cudf,
    )
except ImportError:
    pass
