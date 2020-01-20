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
from cudf.core.buffer import Buffer
from cudf.utils.cudautils import zeros
from cudf.utils.dtypes import is_categorical_dtype

from cudf._lib.cudf cimport *
from cudf._lib.cudf import *
from cudf._lib.nvtx import nvtx_range_pop
from cudf._lib.utils cimport *
from cudf._lib.utils import *

cimport cudf._lib.includes.groupby.hash as hash_groupby
cimport cudf._lib.includes.groupby.sort as sort_groupby

agg_names = {
    'sum': hash_groupby.SUM,
    'min': hash_groupby.MIN,
    'max': hash_groupby.MAX,
    'count': hash_groupby.COUNT,
    'mean': hash_groupby.MEAN
}


def groupby(
    keys,
    values,
    ops,
    method='hash',
    sort_results=True,
    dropna=True
):
    """
    Apply aggregations *ops* on *values*, grouping by *keys*.

    Parameters
    ----------
    keys : list of Columns
    values : list of Columns
    ops : str or list of str
        Aggregation to be performed for each column in *values*
    dropna : bool
        Whether or not to drop null keys
    Returns
    -------
    result : tuple of list of Columns
        keys and values of the result
    """
    from cudf.core.column import build_column, build_categorical_column

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

    cdef bool ignore_null_keys = dropna
    cdef hash_groupby.Options *options = new hash_groupby.Options(dropna)

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
        if is_categorical_dtype(inp_key_col.dtype):
            result_key_cols[i] = build_categorical_column(
                categories=inp_key_col.cat().categories,
                codes=result_key_cols[i],
                mask=result_key_cols[i].mask,
                ordered=inp_key_col.cat().ordered
            )

    return (result_key_cols, result_value_cols)


def groupby_without_aggregations(cols, key_cols):
    """
    Sorts the Columns ``cols`` based on the subset ``key_cols``.
    Parameters
    ----------
    cols : list
        List of Columns to be sorted
    key_cols : list
        Subset of *cols* to sort by
    Returns
    -------
    sorted_columns : list
    offsets : Column
        Integer offsets to the start of each set of unique keys
    """
    from cudf.core.column import build_column, build_categorical_column

    cdef cudf_table* c_in_table = table_from_columns(cols)
    cdef vector[size_type] c_key_col_indices
    cdef pair[cudf_table, gdf_column] c_result

    for i in range(len(key_cols)):
        if key_cols[i] in cols:
            c_key_col_indices.push_back(cols.index(key_cols[i]))

    cdef size_t c_num_key_cols = c_key_col_indices.size()

    cdef gdf_context* c_ctx = create_context_view(
        0,
        'sort',
        0,
        0,
        0,
        'null_as_largest',
        True
    )

    with nogil:
        c_result = sort_groupby.gdf_group_by_without_aggregations(
            c_in_table[0],
            c_num_key_cols,
            c_key_col_indices.data(),
            c_ctx
        )

    offsets = gdf_column_to_column(&c_result.second)
    sorted_cols = columns_from_table(&c_result.first)

    for i, inp_col in enumerate(cols):
        if is_categorical_dtype(inp_col.dtype):
            sorted_cols[i] = build_categorical_column(
                categories=inp_col.cat().categories,
                codes=sorted_cols[i],
                mask=sorted_cols[i].mask,
                ordered=inp_col.cat().ordered
            )

    del c_in_table

    return sorted_cols, offsets
