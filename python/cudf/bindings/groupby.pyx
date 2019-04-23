# Copyright (c) 2018, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

# Copyright (c) 2018, NVIDIA CORPORATION.

from cudf.bindings.cudf_cpp cimport *
from cudf.bindings.cudf_cpp import *
from libc.stdint cimport uintptr_t
from libcpp.vector cimport vector

import numpy as np
import pandas as pd
import pyarrow as pa

from librmm_cffi import librmm as rmm
import nvcategory
import nvstrings

cimport cython


# _NAMED_FUNCTIONS = {
#     'mean': gdf_group_by_avg,
#     'min': gdf_group_by_min,
#     'max': gdf_group_by_max,
#     'count': gdf_group_by_count,
#     'sum': gdf_group_by_sum,
# }

cdef void* get_cpp_function(func_name):
    if func_name == 'mean':
        return <void*>gdf_group_by_avg
    elif func_name == 'min':
        return <void*>gdf_group_by_min
    elif func_name == 'max':
        return <void*>gdf_group_by_max
    elif func_name == 'count':
        return <void*>gdf_group_by_count
    elif func_name == 'sum':
        return <void*>gdf_group_by_sum
    else:
        return <void*>NULL


cdef _apply_agg(groupby_class, agg_type, result, add_col_values,
                gdf_context* ctx, val_columns, val_columns_out, sort_result=True):
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
        sort_result : bool
            Boolean indicating whether to sort the results or not
        """

    if sort_result:
        ctx.flag_sort_result = 1

    ncols = len(groupby_class._by)
    # cols = [self._df[thisBy]._column.cffi_view for thisBy in self._by]
    cdef vector[gdf_column*] vector_cols
    # Each of these `column_view_from_column` need to be freed or change cudf_cpp.pyx to use `unique_ptr`
    for thisBy in groupby_class._by:
        vector_cols.push_back(column_view_from_column(groupby_class._df[thisBy]._column))
    # cols = [column_view_from_column(groupby_class._df[thisBy]._column) for thisBy in groupby_class._by]
    # vector_cols = cols

    first_run = add_col_values
    need_to_index = groupby_class._as_index

    col_count = 0
    if isinstance(val_columns, (str, Number)):
        val_columns = [val_columns]

    cdef gdf_column* out_col_indices = NULL
    cdef vector[gdf_column*] vector_out_col_values
    cdef void* agg_func = NULL
    for val_col in val_columns:
        # col_agg = groupby_class._df[val_col]._column.cffi_view
        # Need to free this or change cudf_cpp.pyx to use `unique_ptr`
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
            # out_col_indices = out_col_indices_series._column.cffi_view
            # Need to free this or change cudf_cpp.pyx to use `unique_ptr`
            out_col_indices = column_view_from_column(out_col_indices_series._column)
        # else:
        #     # out_col_indices = ffi.NULL
        #     out_col_indices = NULL

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
        # out_col_values = [
        #     out_col_values_series[i]._column.cffi_view
        #     for i in range(0, ncols)]
        # Each of these `column_view_from_column` need to be freed or change cudf_cpp.pyx to use `unique_ptr`
        for i in range(0, ncols):
            vector_out_col_values.push_back(column_view_from_column(out_col_values_series[i]._column))
        # out_col_values = [
        #     column_view_from_column(out_col_values_series[i]._column)
        #     for i in range(0, ncols)]
        # vector_out_col_values = out_col_values

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

        # out_col_agg = out_col_agg_series._column.cffi_view
        # Need to free this or change cudf_cpp.pyx to use `unique_ptr`
        out_col_agg = column_view_from_column(out_col_agg_series._column)

        # agg_func = _NAMED_FUNCTIONS.get(agg_type, None)
        # if agg_func is None:
        #     raise RuntimeError(
        #         "ERROR: this aggregator has not been implemented yet"
        #     )
        agg_func = get_cpp_function(agg_type)
        if agg_func is NULL:
            raise RuntimeError(
                "ERROR: this aggregator has not been implemented yet"
            )


        # err = agg_func(
        #     ncols,
        #     cols,
        #     col_agg,
        #     out_col_indices,
        #     out_col_values,
        #     out_col_agg,
        #     ctx)

        with nogil:
            err = agg_func(
                ncols, #done
                vector_cols.data(), #done
                col_agg, #done
                out_col_indices, #done
                vector_out_col_values.data(), #done
                out_col_agg, #done
                ctx #done
            )

        check_gdf_error(err)

        num_row_results = out_col_agg.size

        # NVStrings columns are not the same going in as coming out but we
        # can't create entire memory views otherwise multiple objects will
        # try to free the memory
        for i, col in enumerate(out_col_values_series):
            if col.dtype == np.dtype("object") and len(col) > 0:
                import nvcategory
                # nvcat_ptr = int(
                #     ffi.cast(
                #         "uintptr_t",
                #         out_col_values[i].dtype_info.category
                #     )
                # )
                nvcat_ptr = int(<uintptr_t>out_col_values[i].dtype_info.category)
                nvcat_obj = None
                if nvcat_ptr:
                    nvcat_obj = nvcategory.bind_cpointer(nvcat_ptr)
                    nvstr_obj = nvcat_obj.to_strings()
                else:
                    import nvstrings
                    nvstr_obj = nvstrings.to_device([])
                out_col_values_series[i]._column._data = nvstr_obj
                out_col_values_series[i]._column._nvcategory = nvcat_obj
        if out_col_agg_series.dtype == np.dtype("object") and \
                len(out_col_agg_series) > 0:
            import nvcategory
            # nvcat_ptr = int(
            #     ffi.cast(
            #         "uintptr_t",
            #         out_col_agg.dtype_info.category
            #     )
            # )
            nvcat_ptr = int(<uintptr_t>out_col_agg.dtype_info.category)
            nvcat_obj = None
            if nvcat_ptr:
                nvcat_obj = nvcategory.bind_cpointer(nvcat_ptr)
                nvstr_obj = nvcat_obj.to_strings()
            else:
                import nvstrings
                nvstr_obj = nvstrings.to_device([])
            out_col_agg_series._column._data = nvstr_obj
            out_col_agg_series._column._nvcategory = nvcat_obj

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

    return result
