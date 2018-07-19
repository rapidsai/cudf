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

from libgdf_cffi import ffi, libgdf, GDFError

import pytest


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
       ncols = len(self._by)
       cols = [self._df[thisBy]._column.cffi_view for thisBy in self._by]
       
       col_agg = self._df[self._val_columns[0]]._column.cffi_view
       
       out_col_indices = ffi.new('gdf_column*', None)
       out_col_values = ffi.new('gdf_column**', None)
       out_col_agg = ffi.new('gdf_column*')
       
       out_col_agg.dtype = col_agg.dtype
       
       ctx = ffi.new('gdf_context*')
       ctx.flag_sorted = 1
       ctx.flag_method = libgdf.GDF_SORT
              
       err = libgdf.gdf_group_by_avg(ncols, cols, col_agg, out_col_indices, out_col_values, out_col_agg, ctx)
       
       print(err)
       
       print("done mean")
       
       
       
       
       """
       wsm todo
       
       i think we can access the columns of a data frame and turn them into gdf_columns
       df[col]._column.cffi_view  something like that
       
       so we need to take the columns from the df that we are grouping by and place into a gdf_column ** (HARD?)
       then we need to take the column(s) to agg and make into a gdf_column* (EASY?)
       and collect the result
       how to we print or assert?? 
       
       
       """
       
       
