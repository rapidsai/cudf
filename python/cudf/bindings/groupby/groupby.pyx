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


def apply_groupby(keys, values, ops, method='hash'):
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

    for i, inp_value_col in enumerate(values):
         if isinstance(inp_value_col, CategoricalColumn):
             result_value_cols[i] = CategoricalColumn(
                 data=result_value_cols[i].data,
                 mask=result_value_cols[i].mask,
                 categories=inp_value_col.cat().categories,
                 ordered=inp_value_col.cat().ordered
             )

    return (result_key_cols, result_value_cols)


def agg(groupby_class, args):
    """ Invoke aggregation functions on the groups.

    Parameters
    ----------
    groupby_class : :class:`~cudf.groupby.Groupby`
        Instance of :class:`~cudf.groupby.Groupby`.
    args : dict, list, str, callable
        - str
            The aggregate function name.
        - list
            List of *str* of the aggregate function.
        - dict
            key-value pairs of source column name and list of
            aggregate functions as *str*.

    Returns
    -------
    result : DataFrame
    """
    sort_results = True

    key_columns = []
    value_columns = []
    result_key_names = []
    result_value_names = []
    agg_names = []

    for colname in groupby_class._by:
        key_columns.append(groupby_class._df._cols[colname]._column)
        result_key_names = groupby_class._by

    use_prefix = (
        len(groupby_class._val_columns) > 1
        or len(args) > 1
    )
    if not isinstance(args, str) and isinstance(
            args, collections.abc.Sequence):
        # call apply_groupby and then rename the columns
        for colname in groupby_class._val_columns:
            in_column = groupby_class._df._cols[colname]._column
            for agg_type in args:
                value_columns.append(in_column)
                if use_prefix:
                    result_value_names.append(
                        agg_type + '_' + colname)
                agg_names.append(agg_type)
            if not use_prefix:
                result_value_names.extend(groupby_class._val_columns)

        result_keys, result_values = apply_groupby(
            key_columns,
            value_columns,
            agg_names
        )

        result = cudf.concat(
            [
                dataframe_from_columns(result_keys, columns=result_key_names),
                dataframe_from_columns(result_values, columns=result_value_names)
            ],
            axis=1
        )

        if (
                result_key_names == result_value_names
                and len(result_key_names) == 1
        ):
            # Special case as index and column have the same name,
            # which `concat` cannot deal with
            result = cudf.DataFrame(
                {result_key_names[0]: result_values[0]},
                index=cudf.Series(result_keys[0], name=result_key_names[0])
            )
            if sort_results:
                result = result.sort_index()
            return result

        if sort_results:
            result = result.sort_values(result_key_names)

        if(groupby_class._as_index):
            result = groupby_class.apply_multiindex_or_single_index(result)
        if use_prefix and groupby_class._as_index:
            result = groupby_class.apply_multicolumn(result, args)

    elif isinstance(args, collections.abc.Mapping):
        if len(args) == 1:
            for key, value in args.items():
                if isinstance(value, str):
                    use_prefix = False
                else:
                    if len(value) == 1:
                        use_prefix = False
        for colname, agg_type in args.items():
            in_column = groupby_class._df._cols[colname]._column
            if not isinstance(agg_type, str) and \
                    isinstance(agg_type, collections.abc.Sequence):
                for sub_agg_type in agg_type:
                    agg_names.append(sub_agg_type)
                    value_columns.append(in_column)
                    if use_prefix:
                        result_value_names.append(
                            sub_agg_type + '_' + colname
                        )
                    if not use_prefix:
                        result_value_names.append(colname)
            elif isinstance(agg_type, str):
                value_columns.append(in_column)
                agg_names.append(agg_type)
                if use_prefix:
                    result_value_names.append(
                        agg_type + '_' + colname
                    )
                if not use_prefix:
                    result_value_names.append(colname)

        result_keys, result_values = apply_groupby(
            key_columns,
            value_columns,
            agg_names
        )

        result = cudf.concat(
            [
                dataframe_from_columns(result_keys, columns=result_key_names),
                dataframe_from_columns(result_values, columns=result_value_names)
            ],
            axis=1
        )

        if (
                result_key_names == result_value_names
                and len(result_key_names) == 1
        ):
            # Special case as index and column have the same name,
            # which `concat` cannot deal with
            result = cudf.DataFrame(
                {result_key_names[0]: result_values[0]},
                index=cudf.Series(result_keys[0], name=result_key_names[0])
            )
            if sort_results:
                result = result.sort_index()
            return result

        if sort_results:
            result = result.sort_values(result_key_names)

        if groupby_class._as_index:
            result = groupby_class.apply_multiindex_or_single_index(result)
        if use_prefix and groupby_class._as_index:
            result = groupby_class.apply_multicolumn_mapped(result, args)
    else:
        result = groupby_class.agg([args])

    nvtx_range_pop()
    return result


def apply_basic_agg(groupby_class, agg_type, sort_results=False):
    """
    Parameters
    ----------
    groupby_class : :class:`~cudf.groupby.Groupby`
        Instance of :class:`~cudf.groupby.Groupby`.
    agg_type : str
        The aggregation function to run.
    """
    sort_results = True

    key_columns = columns_from_dataframe(groupby_class._df[groupby_class._by])
    value_columns = columns_from_dataframe(groupby_class._df[groupby_class._val_columns])
    result_key_names = groupby_class._by
    result_value_names = groupby_class._val_columns

    result_keys, result_values = apply_groupby(
        key_columns,
        value_columns,
        agg_type
    )

    result = cudf.concat(
        [
            dataframe_from_columns(result_keys, columns=result_key_names),
            dataframe_from_columns(result_values, columns=result_value_names)
        ],
        axis=1
    )

    if (
            result_key_names == result_value_names
            and len(result_key_names) == 1
    ):
        # Special case as index and column have the same name,
        # which `concat` cannot deal with
        result = cudf.DataFrame(
            {result_key_names[0]: result_values[0]},
            index=cudf.Series(result_keys[0], name=result_key_names[0])
        )
        if sort_results:
            result = result.sort_index()
        return result

    if sort_results:
        result = result.sort_values(result_key_names)

    # If a Groupby has one index column and one value column
    # and as_index is set, return a Series instead of a df
    if isinstance(result_value_names, (str, Number)) and groupby_class._as_index:
        result_series = result[result_value_names]
        idx = index.as_index(result[groupby_class._by[0]])
        if groupby_class.level == 0:
            idx.name = groupby_class._original_index_name
        else:
            idx.name = groupby_class._by[0]
        result_series = result_series.set_index(idx)
        if groupby_class._as_index:
            result = groupby_class.apply_multiindex_or_single_index(result)
            result_series.index = result.index
        return result_series

    if groupby_class._as_index:
        result = groupby_class.apply_multiindex_or_single_index(result)

    nvtx_range_pop()

    return result
