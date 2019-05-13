import numbers

import numpy as np
import pandas as pd


import cudf
from cudf.dataframe.series import Series


class Indexer(object):
    """
    Base class providing utilities for Loc/ILoc indexers
    """
    def __getitem__(self, arg):

        if type(arg) is not tuple:
            arg = (arg, self._df.columns)

        if self._is_multiindex_arg(arg):
            return self._getitem_multiindex_arg(arg)
        elif self._is_scalar_access(arg):
            return self._getitem_scalar(arg)

        df = self._getitem_tuple_arg(arg)

        if self._can_downcast_to_series(df):
            return self._downcast_to_series(df)

        return df

    def _normalize_dtypes(self, df):
        dtypes = df.dtypes.values.tolist()
        normalized_dtype = np.result_type(*dtypes)
        for name, col in df.iteritems():
            df[name] = col.astype(normalized_dtype)
        return df

    def _is_single_value(self, val):
        from pandas.core.dtypes.common import is_datetime_or_timedelta_dtype
        if (
                isinstance(val, str)
                or isinstance(val, numbers.Number)
                or is_datetime_or_timedelta_dtype(val)
                or isinstance(val, pd.Timestamp)
                or isinstance(val, pd.Categorical)
        ):
            return True
        return False


class Loc(Indexer):
    """
    For selection by label.
    """

    def __init__(self, df):
        self._df = df

    def _getitem_tuple_arg(self, arg):
        from cudf.dataframe.dataframe import DataFrame
        from cudf.dataframe.index import as_index
        columns = self._get_column_selection(arg[1])
        df = DataFrame()
        for col in columns:
            df.add_column(name=col, data=self._df[col].loc[arg[0]])
        if df.shape[0] == 1: # we have a single row without an index
            df.index = as_index(arg[0])
        return df

    def _getitem_scalar(self, arg):
        return self._df[arg[1]].loc[arg[0]]

    def _getitem_multiindex_arg(self, arg):
        # Explicitly ONLY support tuple indexes into MultiIndex.
        # Pandas allows non tuple indices and warns "results may be
        # undefined."
        return self._df._index._get_row_major(self._df, arg)

    def _get_column_selection(self, arg):
        if self._is_single_value(arg):
            return [arg]
        elif isinstance(arg, slice):
            start, stop, step = arg.indices(self._df.shape[1])
            cols = []
            for i in range(start, stop, step):
                cols.append(self._df.columns[i])
            return cols
        else:
            return arg

    def _is_scalar_access(self, arg):
        if isinstance(arg, str):
            return False
        if not hasattr(arg, '__len__'):
            return False
        for obj in arg:
            if not self._is_single_value(obj):
                return False
        return True

    def _is_multiindex_arg(self, arg):
        return (
            isinstance(self._df.index, cudf.dataframe.multiindex.MultiIndex)
            and isinstance(arg, tuple)
        )

    def _can_downcast_to_series(self, df):
        nrows, ncols = df.shape
        if nrows == 1:
            dtypes = df.dtypes.values.tolist()
            all_numeric = all([pd.api.types.is_numeric_dtype(t) for t in dtypes])
            all_identical = dtypes.count(dtypes[0]) == len(dtypes)
            if all_numeric or all_identical:
                return True
        if ncols == 1:
            return True
        return False

    def _downcast_to_series(self, df):
        if df.shape[0] == 1:
            indices = df.columns
            df = self._normalize_dtypes(df)
            sr = df.T
            return sr[sr.columns[0]]

        if df.shape[1] == 1:
            return df[df.columns[0]]


class Iloc(object):
    """
    For integer-location based selection.
    """

    def __init__(self, df):
        self._df = df

    def __getitem__(self, arg):
        if isinstance(arg, (tuple)):
            if len(arg) == 1:
                arg = list(arg)
            elif len(arg) == 2:
                return self[arg[0]][arg[1]]
            else:
                return pd.core.indexing.IndexingError(
                    "Too many indexers"
                )

        if isinstance(arg, numbers.Integral):
            rows = []
            for col in self._df.columns:
                rows.append(self._df[col][arg])
            return Series(np.array(rows), name=arg)
        else:
            from cudf.dataframe.dataframe import DataFrame
            df = DataFrame()
            for col in self._df.columns:
                df[col] = self._df[col][arg]
            return df

    def __setitem__(self, key, value):
        # throws an exception while updating
        msg = "updating columns using iloc is not allowed"
        raise ValueError(msg)
