# Copyright (c) 2018, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

# Copyright (c) 2018, NVIDIA CORPORATION.


import collections
import numpy as np
from numbers import Number

from cython.operator cimport dereference as deref
from libcpp.utility cimport pair
from libcpp.vector cimport vector
from libc.stdlib cimport malloc, free

import cudf
import cudf.dataframe.index as index
from cudf.bindings.cudf_cpp cimport *
from cudf.bindings.cudf_cpp import *
from cudf.dataframe.dataframe import DataFrame
from cudf.dataframe.series import Series
from cudf.dataframe.buffer import Buffer
from cudf.dataframe.categorical import CategoricalColumn
from cudf.utils.cudautils import zeros
from cudf.bindings.nvtx import nvtx_range_pop
from cudf.bindings.utils cimport *
from cudf.bindings.utils import *
cimport cudf.bindings.groupby.hash as hash_groupby


agg_names = {
    'sum': hash_groupby.SUM,
    'min': hash_groupby.MIN,
    'max': hash_groupby.MAX,
    'count': hash_groupby.COUNT,
    'mean': hash_groupby.MEAN
}


def columns_from_dataframe(df):
    return [sr._column for sr in df._cols.values()]


def dataframe_from_columns(cols, index=None, columns=None):
    df = cudf.DataFrame(dict(zip(range(len(cols)), cols)), index=index)
    if columns is not None:
        df.columns = columns
    return df


def apply_groupby(keys, values, ops, method='hash', sort_results=True):
    """
    Apply aggregations *ops* on *values*, grouping by *keys*.

    Parameters
    ----------
    keys : list of Columns
    values : list of Columns
    ops : str or list of str
        Aggregation to be performed for each column in *values*

    Returns
    -------
    result : tuple of list of Columns
        keys and values of the result
    """
    if len(values) == 0:
        return (keys, [])

    cdef pair[cudf_table, cudf_table] result
    cdef cudf_table *c_keys_table = table_from_columns(keys)
    cdef cudf_table *c_values_table = table_from_columns(values)
    cdef vector[hash_groupby.operators] c_ops

    num_values_cols = len(values)
    for i in range(num_values_cols):
        if isinstance(ops, str):
            c_ops.push_back(agg_names[ops])
        else:
            c_ops.push_back(agg_names[ops[i]])

    cdef hash_groupby.Options *options = new hash_groupby.Options()

    with nogil:
        result = hash_groupby.groupby(
            c_keys_table[0],
            c_values_table[0],
            c_ops,
            deref(options)
        )

    del c_keys_table
    del c_values_table
    del options

    result_key_cols = columns_from_table(&result.first)
    result_value_cols = columns_from_table(&result.second)

    for i, inp_key_col in enumerate(keys):
        if isinstance(inp_key_col, CategoricalColumn):
            result_key_cols[i] = CategoricalColumn(
                data=result_key_cols[i].data,
                mask=result_key_cols[i].mask,
                categories=inp_key_col.cat().categories,
                ordered=inp_key_col.cat().ordered
            )

    return (result_key_cols, result_value_cols)
