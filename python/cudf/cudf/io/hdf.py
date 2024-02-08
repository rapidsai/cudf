# Copyright (c) 2019-2024, NVIDIA CORPORATION.

import warnings

import pandas as pd

import cudf
from cudf.utils import ioutils


@ioutils.doc_read_hdf()
def read_hdf(path_or_buf, *args, **kwargs):
    """{docstring}"""
    warnings.warn(
        "Using CPU via Pandas to read HDF dataset, this may "
        "be GPU accelerated in the future"
    )
    pd_value = pd.read_hdf(path_or_buf, *args, **kwargs)
    return cudf.from_pandas(pd_value)


@ioutils.doc_to_hdf()
def to_hdf(path_or_buf, key, value, *args, **kwargs):
    """{docstring}"""
    warnings.warn(
        "Using CPU via Pandas to write HDF dataset, this may "
        "be GPU accelerated in the future"
    )
    pd_value = value.to_pandas()
    pd_value.to_hdf(path_or_buf, key=key, *args, **kwargs)
