# Copyright (c) 2018, NVIDIA CORPORATION.

import numpy as np

from numba import cuda

from .dataframe import DataFrame, Series
from .buffer import Buffer

from libgdf_cffi import ffi, libgdf, GDFError


# import pytest

print("class")


class LibGdfGroupby(object):
    """Groupby object returned by pygdf.DataFrame.groupby().
    """

    def __init__(self, df, by, method="GDF_SORT"):
        """
        Parameters
        ----------
        df : DataFrame
        by : str of list of str
            Column(s) that grouping is based on.
            It can be a single or list of column names.
        """
        print("__init__")

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

        if (val_columns_out is None):
            val_columns_out = val_columns

        ncols = len(self._by)
        cols = [self._df[thisBy]._column.cffi_view for thisBy in self._by]

        first_run = add_col_values
        need_to_index = len(
            self._val_columns) > 0 and self._method == libgdf.GDF_HASH

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

            if first_run:
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

            if agg_type == "min":
                err = libgdf.gdf_group_by_min(
                    ncols, cols, col_agg, out_col_indices, out_col_values, out_col_agg, ctx)
            elif agg_type == "max":
                err = libgdf.gdf_group_by_max(
                    ncols, cols, col_agg, out_col_indices, out_col_values, out_col_agg, ctx)
            elif agg_type == "count":
                err = libgdf.gdf_group_by_count(
                    ncols, cols, col_agg, out_col_indices, out_col_values, out_col_agg, ctx)
            elif agg_type == "sum":
                err = libgdf.gdf_group_by_sum(
                    ncols, cols, col_agg, out_col_indices, out_col_values, out_col_agg, ctx)
            elif agg_type == "mean":
                err = libgdf.gdf_group_by_avg(
                    ncols, cols, col_agg, out_col_indices, out_col_values, out_col_agg, ctx)
            else:
                print("ERROR: this aggregator has not been implemented yet")

            if (err is not None):
                print(err)

            num_row_results = out_col_agg.size

            if first_run:
                for i in range(0, ncols):
                    out_col_values_series[i].data.size = num_row_results
                    out_col_values_series[i] = out_col_values_series[i].reset_index(
                    )
                    result[self._by[i]] = out_col_values_series[i]

            out_col_agg_series.data.size = num_row_results
            out_col_agg_series = out_col_agg_series.reset_index()

            if need_to_index:
                out_col_indices_series.data.size = num_row_results
                out_col_indices_series = out_col_indices_series.reset_index()
                # TODO do something with the indices to align data

            result[val_columns_out[col_count]] = out_col_agg_series

            out_col_agg_series.data.size = num_row_results
            out_col_agg_series = out_col_agg_series.reset_index()

            first_run = False
            col_count = col_count + 1

        return result

    def min(self):
        agg_type = "min"

        result = DataFrame()
        add_col_values = True

        ctx = ffi.new('gdf_context*')
        ctx.flag_sorted = 0
        ctx.flag_method = self._method
        ctx.flag_distinct = 0

        val_columns = self._val_columns

        return self._apply_agg(
            agg_type, result, add_col_values, ctx, val_columns)

    def max(self):
        agg_type = "max"

        result = DataFrame()
        add_col_values = True

        ctx = ffi.new('gdf_context*')
        ctx.flag_sorted = 0
        ctx.flag_method = self._method
        ctx.flag_distinct = 0

        val_columns = self._val_columns

        return self._apply_agg(
            agg_type, result, add_col_values, ctx, val_columns)

    def count(self):
        agg_type = "count"

        result = DataFrame()
        add_col_values = True

        ctx = ffi.new('gdf_context*')
        ctx.flag_sorted = 0
        ctx.flag_method = self._method
        ctx.flag_distinct = 0

        val_columns = self._val_columns

        return self._apply_agg(
            agg_type, result, add_col_values, ctx, val_columns)

    def sum(self):
        agg_type = "sum"

        result = DataFrame()
        add_col_values = True

        ctx = ffi.new('gdf_context*')
        ctx.flag_sorted = 0
        ctx.flag_method = self._method
        ctx.flag_distinct = 0

        val_columns = self._val_columns

        return self._apply_agg(
            agg_type, result, add_col_values, ctx, val_columns)

    def mean(self):
        agg_type = "mean"

        result = DataFrame()
        add_col_values = True

        ctx = ffi.new('gdf_context*')
        ctx.flag_sorted = 0
        ctx.flag_method = self._method
        ctx.flag_distinct = 0

        val_columns = self._val_columns

        return self._apply_agg(
            agg_type, result, add_col_values, ctx, val_columns)

    def agg(self, args):

        result = DataFrame()
        add_col_values = True

        ctx = ffi.new('gdf_context*')
        ctx.flag_sorted = 0
        ctx.flag_method = self._method
        ctx.flag_distinct = 0

        if isinstance(args, (tuple, list)):
            for agg_type in args:

                val_columns_out = [val + '_' +
                                   agg_type for val in self._val_columns]

                result = self._apply_agg(
                    agg_type, result, add_col_values, ctx, self._val_columns, val_columns_out)

                add_col_values = False  # we only want to add them once

        elif isinstance(args, dict):
            for val, agg_type in args.items():

                val_columns_out = [val + '_' + agg_type]

                result = self._apply_agg(agg_type, result, add_col_values, ctx, [
                                         val], val_columns_out)

                add_col_values = False  # we only want to add them once

        else:
            result = self.agg([args])

        return result
