# Copyright (c) 2018, NVIDIA CORPORATION.

from collections import OrderedDict, defaultdict, namedtuple

from itertools import chain
import numpy as np

from numba import cuda

from .dataframe import DataFrame, Series
from .multi import concat
from . import _gdf, cudautils
from .column import Column
from .buffer import Buffer
from .serialize import register_distributed_serializer
from .index import RangeIndex

from libgdf_cffi import ffi, libgdf, GDFError



# import pytest

print("class")


class LibGdfGroupby(object):
    """Groupby object returned by pygdf.DataFrame.groupby().
    """

    def __init__(self, df, by):
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

    def mean(self):
        #          """
        #         gdf_error gdf_group_by_avg(int ncols,                    // # columns
        #                            gdf_column** cols,            //input cols
        #                            gdf_column* col_agg,          //column to aggregate on
        #                            gdf_column* out_col_indices,  //if not null return indices of re-ordered rows
        #                            gdf_column** out_col_values,  //if not null return the grouped-by columns
        #                                                          //(multi-gather based on indices, which are needed anyway)
        #                            gdf_column* out_col_agg,      //aggregation result
        #                            gdf_context* ctxt);            //struct with additional info: bool is_sorted, flag_sort_or_hash, bool flag_count_distinct
        #        """
        print("start mean")

        result = DataFrame()
        
        indexes = DataFrame()
        
        ctx = ffi.new('gdf_context*')
        ctx.flag_sorted = 0
        ctx.flag_method = libgdf.GDF_SORT
        ctx.flag_distinct = 0

        ncols = len(self._by)
        cols = [self._df[thisBy]._column.cffi_view for thisBy in self._by]
        
        first_run = True
        multiple_aggs = len(self._val_columns) > 0
        
        for val_col in self._val_columns:
            col_agg = self._df[val_col]._column.cffi_view
            col_agg_dtype = self._df[val_col]._column.data.dtype
            
            #  assuming here that if there are multiple aggregations that the aggregated results will be in the same order
            # this may need to be revised for hash group bys. May want to collect indexes to join multiple resulting aggregations
#             if multiple_aggs:
#                 out_col_indices_series = Series(Buffer(cuda.device_array(col_agg.size, dtype=np.int32)))
#                 out_col_indices = out_col_indices_series._column.cffi_view
#             else:
            out_col_indices = ffi.NULL

            if first_run:
                out_col_values_series = [Series(Buffer(cuda.device_array(col_agg.size, dtype=self._df[self._by[i]]._column.data.dtype))) for i in range(0,ncols)]
                out_col_values = [out_col_values_series[i]._column.cffi_view for i in range(0,ncols)]
            else :
                out_col_values = ffi.NULL
    
            out_col_agg_series = Series(Buffer(cuda.device_array(col_agg.size, dtype=col_agg_dtype)))
            out_col_agg = out_col_agg_series._column.cffi_view
    
            err = libgdf.gdf_group_by_avg(
                ncols, cols, col_agg, out_col_indices, out_col_values, out_col_agg, ctx)
    
            print(err)
    
            print("done mean")
            
            num_row_results = out_col_agg.size
            
            if first_run:
                for i in range(0,ncols):
                    out_col_values_series[i].data.size = num_row_results
                    out_col_values_series[i] = out_col_values_series[i].reset_index()
                    result[self._by[i]] = out_col_values_series[i]
    
            out_col_agg_series.data.size = num_row_results
            out_col_agg_series = out_col_agg_series.reset_index()
            
            #  assuming here that if there are multiple aggregations that the aggregated results will be in the same order
            # this may need to be revised for hash group bys
            result[val_col] = out_col_agg_series  
            
            first_run = False
            
        return result
        
   
  
