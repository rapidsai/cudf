# Copyright (c) 2018, NVIDIA CORPORATION.

import numpy as np
import collections

from numbers import Number
from pandas.api.types import is_categorical_dtype

from cudf.dataframe.dataframe import DataFrame
from cudf.dataframe.series import Series
from cudf.dataframe.buffer import Buffer
from cudf.dataframe.categorical import CategoricalColumn
from cudf.utils.cudautils import zeros
from cudf._gdf import nvtx_range_pop
import cudf.dataframe.index as index

from libgdf_cffi import ffi, libgdf
from librmm_cffi import librmm as rmm


class SeriesGroupBy(object):
    """Wraps DataFrameGroupby with special attr methods
    """
    def __init__(self, source_series, group_series, level=None, sort=False):
        self.source_series = source_series
        self.group_series = group_series
        self.level = level
        self.sort = sort

    def __getattr__(self, attr):
        df = DataFrame()
        df['x'] = self.source_series
        if self.level is not None:
            df['y'] = self.source_series.index
        else:
            df['y'] = self.group_series
        groupby = df.groupby('y', level=self.level, sort=self.sort)
        result_df = getattr(groupby, attr)()

        def get_result():
            result_series = result_df['x']
            result_series.name = None
            idx = result_df.index
            idx.name = None
            result_series.set_index(idx)
            return result_series
        return get_result

    def agg(self, agg_types):
        df = DataFrame()
        df['x'] = self.source_series
        if self.level is not None:
            df['y'] = self.source_series.index
        else:
            df['y'] = self.group_series
        groupby = df.groupby('y').agg(agg_types)
        idx = groupby.index
        idx.name = None
        groupby.set_index(idx)
        return groupby


class Groupby(object):
    """Groupby object returned by cudf.DataFrame.groupby().
    """

    _NAMED_FUNCTIONS = {'mean': libgdf.gdf_group_by_avg,
                        'min': libgdf.gdf_group_by_min,
                        'max': libgdf.gdf_group_by_max,
                        'count': libgdf.gdf_group_by_count,
                        'sum': libgdf.gdf_group_by_sum,
                        }

    _LEVEL_0_INDEX_NAME = 'cudf_groupby_level_index'

    def __init__(self, df, by, method="hash", as_index=True, level=None):
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
            group by. Valid values are "hash".
        """
        self.level = None
        self._original_index_name = None
        self._df = df
        if isinstance(by, Series):
            if len(by) != len(self._df.index):
                raise NotImplementedError("CUDF doesn't support series groupby"
                                          "with indices of arbitrary length")
            self.level = 0
            self._df[self._LEVEL_0_INDEX_NAME] = by
            self._original_index_name = self._df.index.name
            self._by = [self._LEVEL_0_INDEX_NAME]
        elif level == 0:
            self.level = level
            self._df[self._LEVEL_0_INDEX_NAME] = self._df.index
            self._original_index_name = self._df.index.name
            self._by = [self._LEVEL_0_INDEX_NAME]
        elif level and level > 0:
            raise NotImplementedError('MultiIndex not supported yet in cudf')
        else:
            self._by = [by] if isinstance(by, (str, Number)) else list(by)
        self._val_columns = [idx for idx in self._df.columns
                             if idx not in self._by]
        self._as_index = as_index
        if (method == "hash"):
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

        if sort_result:
            ctx.flag_sort_result = 1

        ncols = len(self._by)
        cols = [self._df[thisBy]._column.cffi_view for thisBy in self._by]

        first_run = add_col_values
        need_to_index = self._as_index

        col_count = 0
        if isinstance(val_columns, (str, Number)):
            val_columns = [val_columns]
        for val_col in val_columns:
            col_agg = self._df[val_col]._column.cffi_view

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
                out_col_indices = out_col_indices_series._column.cffi_view
            else:
                out_col_indices = ffi.NULL

            out_col_values_series = []
            for i in range(0, ncols):
                if self._df[self._by[i]].dtype == np.dtype('object'):
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
                                dtype=self._df[self._by[i]]._column.data.dtype
                            )
                        )
                    )
                out_col_values_series.append(col)
            out_col_values = [
                out_col_values_series[i]._column.cffi_view
                for i in range(0, ncols)]

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
                if self._df[val_col].dtype == np.dtype('object'):
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
                                dtype=self._df[val_col]._column.data.dtype
                            )
                        )
                    )

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
                raise RuntimeError(err)

            num_row_results = out_col_agg.size

            # NVStrings columns are not the same going in as coming out but we
            # can't create entire CFFI views otherwise multiple objects will
            # try to free the memory
            for i, col in enumerate(out_col_values_series):
                if col.dtype == np.dtype("object") and len(col) > 0:
                    import nvcategory
                    nvcat_ptr = int(
                        ffi.cast(
                            "uintptr_t",
                            out_col_values[i].dtype_info.category
                        )
                    )
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
                nvcat_ptr = int(
                    ffi.cast(
                        "uintptr_t",
                        out_col_agg.dtype_info.category
                    )
                )
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
                for i, thisBy in enumerate(self._by):
                    result[thisBy] = out_col_values_series[i][
                        :num_row_results]

                    if is_categorical_dtype(self._df[thisBy].dtype):
                        result[thisBy] = CategoricalColumn(
                            data=result[thisBy].data,
                            categories=self._df[thisBy].cat.categories,
                            ordered=self._df[thisBy].cat.ordered
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

    def _apply_basic_agg(self, agg_type, sort_results=False):
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
        val_columns_out = self._val_columns

        result = self._apply_agg(
            agg_type, result, add_col_values, ctx, val_columns,
            val_columns_out, sort_result=sort_results)

        # If a Groupby has one index column and one value column
        # and as_index is set, return a Series instead of a df
        if isinstance(val_columns, (str, Number)) and self._as_index:
            result_series = result[val_columns]
            idx = index.as_index(result[self._by[0]])
            if self.level == 0:
                idx.name = self._original_index_name
            else:
                idx.name = self._by[0]
            result_series = result_series.set_index(idx)
            return result_series

        # TODO: Do MultiIndex here
        if(self._as_index):
            idx = index.as_index(result[self._by[0]])
            idx.name = self._by[0]
            result.drop_column(idx.name)
            if self.level == 0:
                idx.name = self._original_index_name
            else:
                idx.name = self._by[0]
            result = result.set_index(idx)

        nvtx_range_pop()

        return result

    def __getitem__(self, arg):
        if isinstance(arg, (str, Number)):
            if arg not in self._val_columns:
                raise KeyError("Column not found: " + str(arg))
        else:
            for val in arg:
                if val not in self._val_columns:
                    raise KeyError("Column not found: " + str(val))
        result = self.copy()
        result._val_columns = arg
        return result

    def copy(self, deep=True):
        df = self._df.copy(deep) if deep else self._df
        result = Groupby(df, self._by)
        result._method = self._method
        result._val_columns = self._val_columns
        result.level = self.level
        result._original_index_name = self._original_index_name
        return result

    def __getattr__(self, key):
        if key != '_val_columns' and key in self._val_columns:
            return self[key]
        raise AttributeError("'Groupby' object has no attribute %r" % key)

    def min(self, sort=True):
        return self._apply_basic_agg("min", sort)

    def max(self, sort=True):
        return self._apply_basic_agg("max", sort)

    def count(self, sort=True):
        return self._apply_basic_agg("count", sort)

    def sum(self, sort=True):
        return self._apply_basic_agg("sum", sort)

    def mean(self, sort=True):
        return self._apply_basic_agg("mean", sort)

    def agg(self, args):
        """ Invoke aggregation functions on the groups.

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

        # TODO: Use MultiColumn here instead of use_prefix
        # use_prefix enables old functionality - prefixing column
        # groupby names since we don't support MultiColumn quite yet
        use_prefix = 1 < len(self._val_columns) or 1 < len(args)
        if not isinstance(args, str) and isinstance(
                args, collections.abc.Sequence):
            for agg_type in args:
                val_columns_out = [agg_type + '_' +
                                   val for val in self._val_columns]
                if not use_prefix:
                    val_columns_out = self._val_columns
                result = self._apply_agg(
                    agg_type, result, add_col_values, ctx, self._val_columns,
                    val_columns_out, sort_result=sort_result)
                add_col_values = False  # we only want to add them once
            # TODO: Do multindex here
            if(self._as_index) and 1 == len(self._by):
                idx = index.as_index(result[self._by[0]])
                idx.name = self._by[0]
                result = result.set_index(idx)
                result.drop_column(idx.name)
        elif isinstance(args, collections.abc.Mapping):
            if (len(args.keys()) == 1):
                if(len(list(args.values())[0]) == 1):
                    sort_result = False
            for val, agg_type in args.items():

                if not isinstance(agg_type, str) and \
                       isinstance(agg_type, collections.abc.Sequence):
                    for sub_agg_type in agg_type:
                        val_columns_out = [sub_agg_type + '_' + val]
                        if not use_prefix:
                            val_columns_out = self._val_columns
                        result = self._apply_agg(sub_agg_type, result,
                                                 add_col_values, ctx, [val],
                                                 val_columns_out,
                                                 sort_result=sort_result)
                elif isinstance(agg_type, str):
                    val_columns_out = [agg_type + '_' + val]
                    if not use_prefix:
                        val_columns_out = self._val_columns
                    result = self._apply_agg(agg_type, result,
                                             add_col_values, ctx, [val],
                                             val_columns_out,
                                             sort_result=sort_result)
                add_col_values = False  # we only want to add them once
            # TODO: Do multindex here
            if(self._as_index) and 1 == len(self._by):
                idx = index.as_index(result[self._by[0]])
                idx.name = self._by[0]
                result = result.set_index(idx)
                result.drop_column(idx.name)
        else:
            result = self.agg([args])

        nvtx_range_pop()
        return result
