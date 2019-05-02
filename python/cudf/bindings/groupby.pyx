# Copyright (c) 2018, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

# Copyright (c) 2018, NVIDIA CORPORATION.

from cudf.bindings.cudf_cpp cimport *
from cudf.bindings.cudf_cpp import *
from libc.stdint cimport uintptr_t
from libc.stdlib cimport free
from libcpp.vector cimport vector
cimport cython

import collections
import numpy as np
from pandas.api.types import is_categorical_dtype
from numbers import Number

from cudf.dataframe.dataframe import DataFrame
from cudf.dataframe.series import Series
from cudf.dataframe.buffer import Buffer
from cudf.dataframe.categorical import CategoricalColumn
from cudf.utils.cudautils import zeros
from cudf.bindings.nvtx import nvtx_range_pop
import cudf.dataframe.index as index
from librmm_cffi import librmm as rmm
import nvcategory
import nvstrings



cdef _apply_agg(groupby_class, agg_type, result, add_col_values,
                gdf_context* ctx, val_columns, val_columns_out,
                sort_results=True):
    """
        Parameters
        ----------
        groupby_class : :class:`~cudf.groupby.Groupby`
            Instance of :class:`~cudf.groupby.Groupby`.
        agg_type : str
            The aggregation function to run.
        result : DataFrame
            The DataFrame to store the result of the aggregation into.
        add_col_values : bool
            Boolean to indicate whether this is the first aggregation being
            run and should add the additional columns' values.
        ctx : gdf_context* C++ object
            Context object to pass information such as if the dataframe
            is sorted and/or which method to use for grouping.
        val_columns : list of *str*
            The list of column names that the aggregation should be performed
            on.
        val_columns_out : list of *str*
            The list of columns names that the aggregation results should be
            output into.
        sort_results : bool
            Boolean indicating whether to sort the results or not.
        """

    if sort_results:
        ctx.flag_sort_result = 1

    ncols = len(groupby_class._by)
    cdef vector[gdf_column*] vector_cols
    for thisBy in groupby_class._by:
        vector_cols.push_back(column_view_from_column(groupby_class._df[thisBy]._column))

    first_run = add_col_values
    need_to_index = groupby_class._as_index

    col_count = 0
    if isinstance(val_columns, (str, Number)):
        val_columns = [val_columns]

    cdef gdf_column* out_col_indices
    cdef vector[gdf_column*] vector_out_col_values
    cdef gdf_error err
    for val_col in val_columns:
        out_col_indices = NULL
        vector_out_col_values.clear()
        err = GDF_CUDA_ERROR
        col_agg = column_view_from_column(groupby_class._df[val_col]._column)

        # assuming here that if there are multiple aggregations that the
        # aggregated results will be in the same order for GDF_SORT method
        if need_to_index:
            out_col_indices_series = Series(
                Buffer(
                    rmm.device_array(
                        col_agg.size,
                        dtype=np.int32
                    )
                )
            )
            out_col_indices = column_view_from_column(out_col_indices_series._column)
        else:
            out_col_indices = NULL

        out_col_values_series = []
        for i in range(0, ncols):
            if groupby_class._df[groupby_class._by[i]].dtype == np.dtype('object'):
                # This isn't ideal, but no better way to create an
                # nvstrings object of correct size
                gather_map = zeros(col_agg.size, dtype='int32')
                col = Series([''], dtype='str')[gather_map]\
                    .reset_index(drop=True)
            else:
                col = Series(
                    Buffer(
                        rmm.device_array(
                            col_agg.size,
                            dtype=groupby_class._df[groupby_class._by[i]]._column.data.dtype
                        )
                    )
                )
            out_col_values_series.append(col)
        for i in range(0, ncols):
            vector_out_col_values.push_back(column_view_from_column(out_col_values_series[i]._column))

        if agg_type == "count":
            out_col_agg_series = Series(
                Buffer(
                    rmm.device_array(
                        col_agg.size,
                        dtype=np.int64
                    )
                )
            )
        elif agg_type == "mean":
            out_col_agg_series = Series(
                Buffer(
                    rmm.device_array(
                        col_agg.size,
                        dtype=np.float64
                    )
                )
            )
        else:
            if groupby_class._df[val_col].dtype == np.dtype('object'):
                # This isn't ideal, but no better way to create an
                # nvstrings object of correct size
                gather_map = zeros(col_agg.size, dtype='int32')
                out_col_agg_series = Series(
                    [''],
                    dtype='str'
                )[gather_map].reset_index(drop=True)
            else:
                out_col_agg_series = Series(
                    Buffer(
                        rmm.device_array(
                            col_agg.size,
                            dtype=groupby_class._df[val_col]._column.data.dtype
                        )
                    )
                )

        out_col_agg = column_view_from_column(out_col_agg_series._column)

        if agg_type is None:
            raise NotImplementedError(
                "this aggregator has not been implemented yet"
            )
        else:
            with nogil:
                if agg_type == 'mean':
                    err = gdf_group_by_avg(
                        ncols,
                        vector_cols.data(),
                        col_agg,
                        out_col_indices,
                        vector_out_col_values.data(),
                        out_col_agg,
                        ctx
                    )
                elif agg_type == 'min':
                    err = gdf_group_by_min(
                        ncols,
                        vector_cols.data(),
                        col_agg,
                        out_col_indices,
                        vector_out_col_values.data(),
                        out_col_agg,
                        ctx
                    )
                elif agg_type == 'max':
                    err = gdf_group_by_max(
                        ncols,
                        vector_cols.data(),
                        col_agg,
                        out_col_indices,
                        vector_out_col_values.data(),
                        out_col_agg,
                        ctx
                    )
                elif agg_type == 'count':
                    err = gdf_group_by_count(
                        ncols,
                        vector_cols.data(),
                        col_agg,
                        out_col_indices,
                        vector_out_col_values.data(),
                        out_col_agg,
                        ctx
                    )
                elif agg_type == 'sum':
                    err = gdf_group_by_sum(
                        ncols,
                        vector_cols.data(),
                        col_agg,
                        out_col_indices,
                        vector_out_col_values.data(),
                        out_col_agg,
                        ctx
                    )

        check_gdf_error(err)

        num_row_results = out_col_agg.size

        # NVStrings columns are not the same going in as coming out but we
        # can't create entire memory views otherwise multiple objects will
        # try to free the memory
        for i, col in enumerate(out_col_values_series):
            if col.dtype == np.dtype("object") and len(col) > 0:
                update_nvstrings_col(
                    out_col_values_series[i]._column,
                    <uintptr_t>vector_out_col_values[i].dtype_info.category
                )
        if out_col_agg_series.dtype == np.dtype("object") and \
                len(out_col_agg_series) > 0:
            update_nvstrings_col(
                out_col_agg_series._column,
                <uintptr_t>out_col_agg.dtype_info.category
            )

        if first_run:
            for i, thisBy in enumerate(groupby_class._by):
                result[thisBy] = out_col_values_series[i][
                    :num_row_results]

                if is_categorical_dtype(groupby_class._df[thisBy].dtype):
                    result[thisBy] = CategoricalColumn(
                        data=result[thisBy].data,
                        categories=groupby_class._df[thisBy].cat.categories,
                        ordered=groupby_class._df[thisBy].cat.ordered
                    )

        if out_col_agg_series.dtype != np.dtype("object"):
            out_col_agg_series.data.size = num_row_results
        out_col_agg_series = out_col_agg_series.reset_index(drop=True)

        if isinstance(val_columns_out, (str, Number)):
            result[val_columns_out] = out_col_agg_series[:num_row_results]
        else:
            result[val_columns_out[col_count]
                    ] = out_col_agg_series[:num_row_results]

        if out_col_agg_series.dtype != np.dtype("object"):
            out_col_agg_series.data.size = num_row_results
        out_col_agg_series = out_col_agg_series.reset_index(drop=True)

        first_run = False
        col_count = col_count + 1

        # Free objects created each iteration
        free(col_agg)
        free(out_col_indices)
        for val in vector_out_col_values:
            free(val)
        free(out_col_agg)

    # Free objects created once
    for val in vector_cols:
        free(val)

    return result

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

    Notes
    -----
    Since multi-indexes aren't supported aggregation results are returned
    in columns using the naming scheme of `aggregation_columnname`.
    """
    result = DataFrame()
    add_col_values = True

    cdef gdf_context* ctx = create_context_view(0, groupby_class._method, 0, 0, 0)

    sort_results = True

    # TODO: Use MultiColumn here instead of use_prefix
    # use_prefix enables old functionality - prefixing column
    # groupby names since we don't support MultiColumn quite yet
    # https://github.com/rapidsai/cudf/issues/483
    use_prefix = 1 < len(groupby_class._val_columns) or 1 < len(args)
    if not isinstance(args, str) and isinstance(
            args, collections.abc.Sequence):
        for agg_type in args:
            val_columns_out = [
                agg_type + '_' + val for val in groupby_class._val_columns
            ]
            if not use_prefix:
                val_columns_out = groupby_class._val_columns
            result = _apply_agg(
                groupby_class,
                agg_type,
                result,
                add_col_values,
                ctx,
                groupby_class._val_columns,
                val_columns_out,
                sort_results=sort_results
            )
            add_col_values = False  # we only want to add them once
        if(groupby_class._as_index):
            result = groupby_class.apply_multiindex_or_single_index(result)
        if use_prefix:
            result = groupby_class.apply_multicolumn(result, args)
    elif isinstance(args, collections.abc.Mapping):
        if (len(args.keys()) == 1):
            if(len(list(args.values())[0]) == 1):
                sort_results = False
        for val, agg_type in args.items():

            if not isinstance(agg_type, str) and \
                    isinstance(agg_type, collections.abc.Sequence):
                for sub_agg_type in agg_type:
                    val_columns_out = [sub_agg_type + '_' + val]
                    if not use_prefix:
                        val_columns_out = groupby_class._val_columns
                    result = _apply_agg(
                        groupby_class,
                        sub_agg_type,
                        result,
                        add_col_values,
                        ctx,
                        [val],
                        val_columns_out,
                        sort_results=sort_results
                    )
            elif isinstance(agg_type, str):
                val_columns_out = [agg_type + '_' + val]
                if not use_prefix:
                    val_columns_out = groupby_class._val_columns
                result = _apply_agg(
                    groupby_class,
                    agg_type,
                    result,
                    add_col_values,
                    ctx,
                    [val],
                    val_columns_out,
                    sort_results=sort_results
                )
            add_col_values = False  # we only want to add them once
        if groupby_class._as_index:
            result = groupby_class.apply_multiindex_or_single_index(result)
        if use_prefix:
            result = groupby_class.apply_multicolumn_mapped(result, args)
    else:
        result = groupby_class.agg([args])

    free(ctx)

    nvtx_range_pop()
    return result

def _apply_basic_agg(groupby_class, agg_type, sort_results=False):
    """
    Parameters
    ----------
    groupby_class : :class:`~cudf.groupby.Groupby`
        Instance of :class:`~cudf.groupby.Groupby`.
    agg_type : str
        The aggregation function to run.
    """
    result = DataFrame()
    add_col_values = True

    cdef gdf_context* ctx = create_context_view(0, groupby_class._method, 0, 0, 0)

    val_columns = groupby_class._val_columns
    val_columns_out = groupby_class._val_columns

    result = _apply_agg(
        groupby_class,
        agg_type,
        result,
        add_col_values,
        ctx,
        val_columns,
        val_columns_out,
        sort_results=sort_results
    )

    free(ctx)

    # If a Groupby has one index column and one value column
    # and as_index is set, return a Series instead of a df
    if isinstance(val_columns, (str, Number)) and groupby_class._as_index:
        result_series = result[val_columns]
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
