# Copyright (c) 2018-2020, NVIDIA CORPORATION.
import os  # isort:skip

# TODO: Remove the following check once we arrive at a solution for #4827
# This is a temporary workaround to unblock internal testing
# related issue: https://github.com/rapidsai/cudf/issues/4827
if "DASK_PARENT" not in os.environ:
    from cudf.utils.gpu_utils import validate_setup

    validate_setup()

import cupy

import rmm

from cudf import core, datasets
from cudf._version import get_versions
from cudf.core import DataFrame, Index, MultiIndex, Series, from_pandas, merge
from cudf.core.dtypes import CategoricalDtype
from cudf.core.groupby import Grouper
from cudf.core.ops import (
    arccos,
    arcsin,
    arctan,
    cos,
    exp,
    floor_divide,
    log,
    logical_and,
    logical_not,
    logical_or,
    remainder,
    sin,
    sqrt,
    tan,
)
from cudf.core.reshape import concat, get_dummies, melt, merge_sorted
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

cupy.cuda.set_allocator(rmm.rmm_cupy_allocator)

__version__ = get_versions()["version"]
del get_versions
