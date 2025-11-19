# SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import warnings

import pandas as pd

from cudf.core.dataframe import from_pandas
from cudf.utils import ioutils


@ioutils.doc_read_hdf()
def read_hdf(path_or_buf, *args, **kwargs):
    """{docstring}"""
    warnings.warn(
        "Using CPU via Pandas to read HDF dataset, this may "
        "be GPU accelerated in the future"
    )
    pd_value = pd.read_hdf(path_or_buf, *args, **kwargs)
    return from_pandas(pd_value)


@ioutils.doc_to_hdf()
def to_hdf(path_or_buf, key, value, *args, **kwargs):
    """{docstring}"""
    warnings.warn(
        "Using CPU via Pandas to write HDF dataset, this may "
        "be GPU accelerated in the future"
    )
    pd_value = value.to_pandas()
    pd_value.to_hdf(path_or_buf, key=key, *args, **kwargs)
