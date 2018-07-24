# Copyright (c) 2018, NVIDIA CORPORATION.

import numpy as np
import collections

from numba import cuda
from .dataframe import DataFrame, Series
from .buffer import Buffer

from libgdf_cffi import ffi, libgdf, GDFError


class LibGdfGroupby(object):
    """Groupby object returned by pygdf.DataFrame.groupby().
    """

    _NAMED_FUNCTIONS = {'mean': libgdf.gdf_group_by_avg,
                        'min': libgdf.gdf_group_by_min,
                        'max': libgdf.gdf_group_by_max,
                        'count': libgdf.gdf_group_by_count,
                        'sum': libgdf.gdf_group_by_sum,
                        }

    def __init__(self, df, by, method="GDF_SORT"):
        """
        Parameters
        ----------
        df : DataFrame
        by : str of list of str
            Column(s) that grouping is based on.
            It can be a single or list of column names.
        """

        self._df = df
        self._by = [by] if isinstance(by, str) else list(by)
        self._val_columns = [idx for idx in self._df.columns
                             if idx not in self._by]
        if (method == "GDF_SORT"):
            self._method = libgdf.GDF_SORT
        else:
            self._method = libgdf.GDF_HASH

    def _apply_agg(self, agg_type, result, add_col_values,
                   ctx, val_columns, val_columns_out=None):

        if (self._method == libgdf.GDF_HASH):
            ctx.flag_sort_result = 1

        if (val_columns_out is None):
            val_columns_out = val_columns

        ncols = len(self._by)
        cols = [self._df[thisBy]._column.cffi_view for thisBy in self._by]

        first_run = add_col_values
        need_to_index = False
#        need_to_index = len(
#            self._val_columns) > 0 and self._method == libgdf.GDF_HASH

        col_count = 0
        for val_col in val_columns:
            col_agg = self._df[val_col]._column.cffi_view

# assuming here that if there are multiple aggregations that the
# aggregated results will be in the same order for GDF_SORT method
            if need_to_index:
                out_col_indices_series = Series(
                    Buffer(cuda.device_array(col_agg.size, dtype=np.int32)))
                out_col_indices = out_col_indices_series._column.cffi_view
            else:
                out_col_indices = ffi.NULL

            if first_run or self._method == libgdf.GDF_HASH:
                out_col_values_series = [Series(Buffer(cuda.device_array(
                    col_agg.size, dtype=self._df[self._by[i]]._column.data.dtype))) for i in range(0, ncols)]
                out_col_values = [
                    out_col_values_series[i]._column.cffi_view for i in range(0, ncols)]
            else:
                out_col_values = ffi.NULL

            if agg_type == "count":
                out_col_agg_series = Series(
                    Buffer(cuda.device_array(col_agg.size, dtype=np.int64)))
            else:
                out_col_agg_series = Series(Buffer(cuda.device_array(
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
                    result[self._by[i]] = out_col_values_series[i][:num_row_results]

            out_col_agg_series.data.size = num_row_results
            out_col_agg_series = out_col_agg_series.reset_index()

#            if need_to_index:
#                out_col_indices_series.data.size = num_row_results
#                out_col_indices_series = out_col_indices_series.reset_index()
            # TODO do something with the indices to align data

            result[val_columns_out[col_count]
                   ] = out_col_agg_series[:num_row_results]

            out_col_agg_series.data.size = num_row_results
            out_col_agg_series = out_col_agg_series.reset_index()

            first_run = False
            col_count = col_count + 1

        return result

    def _apply_basic_agg(self, agg_type):
        result = DataFrame()
        add_col_values = True

        ctx = ffi.new('gdf_context*')
        ctx.flag_sorted = 0
        ctx.flag_method = self._method
        ctx.flag_distinct = 0

        val_columns = self._val_columns

        return self._apply_agg(
            agg_type, result, add_col_values, ctx, val_columns)

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

        result = DataFrame()
        add_col_values = True

        ctx = ffi.new('gdf_context*')
        ctx.flag_sorted = 0
        ctx.flag_method = self._method
        ctx.flag_distinct = 0

        if not isinstance(args, str) and isinstance(
                args, collections.abc.Sequence):
            for agg_type in args:

                # we don't need to change the output column names
                #                val_columns_out = [val + '_' +
                # agg_type for val in self._val_columns]
                val_columns_out = self._val_columns

                result = self._apply_agg(
                    agg_type, result, add_col_values, ctx, self._val_columns, val_columns_out)

                add_col_values = False  # we only want to add them once

        elif isinstance(args, collections.abc.Mapping):
            for val, agg_type in args.items():

                # we don't need to change the output column names
                #                val_columns_out = [val + '_' + agg_type]
                val_columns_out = [val]

                result = self._apply_agg(agg_type, result, add_col_values, ctx, [
                                         val], val_columns_out)

                add_col_values = False  # we only want to add them once

        else:
            result = self.agg([args])

        return result
