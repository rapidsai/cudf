# Copyright (c) 2018, NVIDIA CORPORATION.

import numpy as np
import collections

from .dataframe import DataFrame, Series
from .buffer import Buffer
from ._gdf import nvtx_range_pop

from libgdf_cffi import ffi, libgdf
from librmm_cffi import librmm as rmm


class LibGdfGroupby(object):
    """Groupby object returned by cudf.DataFrame.groupby().
    """

    _NAMED_FUNCTIONS = {'mean': libgdf.gdf_group_by_avg,
                        'min': libgdf.gdf_group_by_min,
                        'max': libgdf.gdf_group_by_max,
                        'count': libgdf.gdf_group_by_count,
                        'sum': libgdf.gdf_group_by_sum,
                        }

    def __init__(self, df, by, method="sort"):
        """
        Parameters
        ----------
        df : DataFrame
        by : str, list
            - str
                The column name to group on.
            - list
                List of *str* of the column names to group on.
        method : str, optional
            A string indicating the libgdf method to use to perform the
            group by. Valid values are "sort", or "hash".
        """

        self._df = df
        self._by = [by] if isinstance(by, str) else list(by)
        self._val_columns = [idx for idx in self._df.columns
                             if idx not in self._by]
        if (method == "sort"):
            self._method = libgdf.GDF_SORT
        elif (method == "hash"):
            self._method = libgdf.GDF_HASH
        else:
            msg = "Method {!r} is not a supported group by method"
            raise NotImplementedError(msg.format(method))

    def _apply_agg(self, agg_type, result, add_col_values,
                   ctx, val_columns, val_columns_out, sort_result=True):
        """
        Parameters
        ----------
        agg_type : str
            The aggregation function to run.
        result : DataFrame
            The DataFrame to store the result of the aggregation into.
        add_col_values : bool
            Boolean to indicate whether this is the first aggregation being
            run and should add the additional columns' values.
        ctx : gdf_context cffi object
            Context object to pass information such as if the dataframe
            is sorted and/or which method to use for grouping.
        val_columns : list of *str*
            The list of column names that the aggregation should be performed
            on.
        val_columns_out : list of *str*
            The list of columns names that the aggregation results should be
            output into.
        """
        if (self._method == libgdf.GDF_HASH and sort_result):
            ctx.flag_sort_result = 1

        ncols = len(self._by)
        cols = [self._df[thisBy]._column.cffi_view for thisBy in self._by]

        first_run = add_col_values
        need_to_index = False

        col_count = 0
        for val_col in val_columns:
            col_agg = self._df[val_col]._column.cffi_view

            # assuming here that if there are multiple aggregations that the
            # aggregated results will be in the same order for GDF_SORT method
            if need_to_index:
                out_col_indices_series = Series(
                    Buffer(rmm.device_array(col_agg.size, dtype=np.int32)))
                out_col_indices = out_col_indices_series._column.cffi_view
            else:
                out_col_indices = ffi.NULL

            if first_run or self._method == libgdf.GDF_HASH:
                out_col_values_series = [Series(Buffer(rmm.device_array(
                    col_agg.size,
                    dtype=self._df[self._by[i]]._column.data.dtype)))
                    for i in range(0, ncols)]
                out_col_values = [
                    out_col_values_series[i]._column.cffi_view
                    for i in range(0, ncols)]
            else:
                out_col_values = ffi.NULL

            if agg_type == "count":
                out_col_agg_series = Series(
                    Buffer(rmm.device_array(col_agg.size, dtype=np.int64)))
            else:
                out_col_agg_series = Series(Buffer(rmm.device_array(
                    col_agg.size, dtype=self._df[val_col]._column.data.dtype)))

            out_col_agg = out_col_agg_series._column.cffi_view

            agg_func = self._NAMED_FUNCTIONS.get(agg_type, None)
            if agg_func is None:
                raise RuntimeError(
                    "ERROR: this aggregator has not been implemented yet")
            err = agg_func(
                ncols,
                cols,
                col_agg,
                out_col_indices,
                out_col_values,
                out_col_agg,
                ctx)

            if (err is not None):
                print(err)
                raise RuntimeError(err)

            num_row_results = out_col_agg.size

            if first_run:
                for i in range(0, ncols):
                    result[self._by[i]] = out_col_values_series[i][
                        :num_row_results]

            out_col_agg_series.data.size = num_row_results
            out_col_agg_series = out_col_agg_series.reset_index()

            result[val_columns_out[col_count]
                   ] = out_col_agg_series[:num_row_results]

            out_col_agg_series.data.size = num_row_results
            out_col_agg_series = out_col_agg_series.reset_index()

            first_run = False
            col_count = col_count + 1

        return result

    def _apply_basic_agg(self, agg_type):
        """
        Parameters
        ----------
        agg_type : str
            The aggregation function to run.
        """
        result = DataFrame()
        add_col_values = True

        ctx = ffi.new('gdf_context*')
        ctx.flag_sorted = 0
        ctx.flag_method = self._method
        ctx.flag_distinct = 0

        val_columns = self._val_columns
        val_columns_out = [agg_type + "_" + column for column in val_columns]

        result = self._apply_agg(
            agg_type, result, add_col_values, ctx, val_columns,
            val_columns_out, sort_result=False)
        nvtx_range_pop()
        return result

    def min(self):
        return self._apply_basic_agg("min")

    def max(self):
        return self._apply_basic_agg("max")

    def count(self):
        return self._apply_basic_agg("count")

    def sum(self):
        return self._apply_basic_agg("sum")

    def mean(self):
        return self._apply_basic_agg("mean")

    def agg(self, args):
        """Invoke aggregation functions on the groups.

        Parameters
        ----------
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

        ctx = ffi.new('gdf_context*')
        ctx.flag_sorted = 0
        ctx.flag_method = self._method
        ctx.flag_distinct = 0

        sort_result = True

        if not isinstance(args, str) and isinstance(
                args, collections.abc.Sequence):
            if (len(args) == 1 and len(self._val_columns) == 1):
                sort_result = False
            for agg_type in args:

                val_columns_out = [agg_type + '_' +
                                   val for val in self._val_columns]

                result = self._apply_agg(
                    agg_type, result, add_col_values, ctx, self._val_columns,
                    val_columns_out, sort_result=sort_result)

                add_col_values = False  # we only want to add them once

        elif isinstance(args, collections.abc.Mapping):
            if (len(args.keys()) == 1):
                if(len(list(args.values())[0]) == 1):
                    sort_result = False
            for val, agg_type in args.items():

                if not isinstance(agg_type, str) and \
                       isinstance(agg_type, collections.abc.Sequence):
                    for sub_agg_type in agg_type:
                        val_columns_out = [sub_agg_type + '_' + val]
                        result = self._apply_agg(sub_agg_type, result,
                                                 add_col_values, ctx, [val],
                                                 val_columns_out,
                                                 sort_result=sort_result)
                elif isinstance(agg_type, str):
                    val_columns_out = [agg_type + '_' + val]
                    result = self._apply_agg(agg_type, result,
                                             add_col_values, ctx, [val],
                                             val_columns_out,
                                             sort_result=sort_result)

                add_col_values = False  # we only want to add them once

        else:
            result = self.agg([args])

        nvtx_range_pop()
        return result
