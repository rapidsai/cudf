import numpy as np
import pandas as pd
from numba.cuda.cudadrv.devicearray import DeviceNDArray

import cudf
from cudf.utils.cudautils import arange
from cudf.utils.dtypes import is_categorical_dtype, is_scalar


def indices_from_labels(obj, labels):
    from cudf.core.column import column

    labels = column.as_column(labels)

    if is_categorical_dtype(obj.index):
        labels = labels.astype("category")
        codes = labels.codes.astype(obj.index._values.codes.dtype)
        labels = column.build_categorical_column(
            categories=labels.dtype.categories,
            codes=codes,
            ordered=labels.dtype.ordered,
        )
    else:
        labels = labels.astype(obj.index.dtype)

    lhs = cudf.DataFrame({}, index=labels)
    rhs = cudf.DataFrame({"_": arange(len(obj))}, index=obj.index)
    return lhs.join(rhs)["_"]


class _SeriesIlocIndexer(object):
    """
    For integer-location based selection.
    """

    def __init__(self, sr):
        self._sr = sr

    def __getitem__(self, arg):
        if isinstance(arg, tuple):
            arg = list(arg)
        return self._sr[arg]

    def __setitem__(self, key, value):
        if isinstance(key, tuple):
            key = list(key)
        self._sr[key] = value


class _SeriesLocIndexer(object):
    """
    Label-based selection
    """

    def __init__(self, sr):
        self._sr = sr

    def __getitem__(self, arg):
        arg = self._loc_to_iloc(arg)
        return self._sr.iloc[arg]

    def __setitem__(self, key, value):
        key = self._loc_to_iloc(key)
        self._sr.iloc[key] = value

    def _loc_to_iloc(self, arg):
        from cudf.core.series import Series
        from cudf.core.index import Index

        if isinstance(
            arg, (list, np.ndarray, pd.Series, range, Index, DeviceNDArray)
        ):
            if len(arg) == 0:
                arg = Series(np.array([], dtype="int32"))
            else:
                arg = Series(arg)
        if isinstance(arg, Series):
            if arg.dtype in [np.bool, np.bool_]:
                return arg
            else:
                return indices_from_labels(self._sr, arg)
        elif is_scalar(arg):
            found_index = self._sr.index.find_label_range(arg, None)[0]
            return found_index
        elif isinstance(arg, slice):
            start_index, stop_index = self._sr.index.find_label_range(
                arg.start, arg.stop
            )
            return slice(start_index, stop_index, arg.step)
        else:
            raise NotImplementedError(
                ".loc not implemented for label type {}".format(
                    type(arg).__name__
                )
            )


class _DataFrameIndexer(object):
    def __getitem__(self, arg):
        from cudf import MultiIndex

        if isinstance(self._df.index, MultiIndex) or isinstance(
            self._df.columns, MultiIndex
        ):
            # This try/except block allows the use of pandas-like
            # tuple arguments into MultiIndex dataframes.
            try:
                return self._getitem_tuple_arg(arg)
            except (TypeError, KeyError, IndexError, ValueError):
                return self._getitem_tuple_arg((arg, slice(None)))
        else:
            if not isinstance(arg, tuple):
                arg = (arg, slice(None))
            return self._getitem_tuple_arg(arg)

    def __setitem__(self, key, value):
        if not isinstance(key, tuple):
            key = (key, slice(None))
        return self._setitem_tuple_arg(key, value)

    def _can_downcast_to_series(self, df, arg):
        """
        This method encapsulates the logic used
        to determine whether or not the result of a loc/iloc
        operation should be "downcasted" from a DataFrame to a
        Series
        """
        from cudf.core.column import as_column

        if isinstance(df, cudf.Series):
            return False
        nrows, ncols = df.shape
        if nrows == 1:
            if type(arg[0]) is slice:
                if not is_scalar(arg[1]):
                    return False
            else:
                # row selection using boolean indexing - never downcasts
                if pd.api.types.is_bool_dtype(as_column(arg[0]).dtype):
                    return False
            dtypes = df.dtypes.values.tolist()
            all_numeric = all(
                [pd.api.types.is_numeric_dtype(t) for t in dtypes]
            )
            if all_numeric:
                return True
        if ncols == 1:
            if type(arg[1]) is slice:
                if not is_scalar(arg[0]):
                    return False
            if isinstance(arg[1], tuple):
                # Multiindex indexing with a slice
                if any(isinstance(v, slice) for v in arg):
                    return False
            return True
        return False

    def _downcast_to_series(self, df, arg):
        """
        "Downcast" from a DataFrame to a Series
        based on Pandas indexing rules
        """
        nrows, ncols = df.shape
        # determine the axis along which the Series is taken:
        if nrows == 1 and ncols == 1:
            if is_scalar(arg[0]) and is_scalar(arg[1]):
                return df[df.columns[0]][0]
            elif not is_scalar(arg[0]):
                axis = 1
            else:
                axis = 0

        elif nrows == 1:
            axis = 0
        elif ncols == 1:
            axis = 1
        else:
            raise ValueError("Cannot downcast DataFrame selection to Series")

        # take series along the axis:
        if axis == 1:
            return df[df._data.names[0]]
        else:
            df = _normalize_dtypes(df)
            sr = df.T
            return sr[sr._data.names[0]]


class _DataFrameLocIndexer(_DataFrameIndexer):
    """
    For selection by label.
    """

    def __init__(self, df):
        self._df = df

    def _getitem_scalar(self, arg):
        return self._df[arg[1]].loc[arg[0]]

    def _getitem_tuple_arg(self, arg):
        from cudf.core.dataframe import Series, DataFrame
        from cudf.core.column import column
        from cudf.core.index import as_index
        from cudf import MultiIndex

        # Step 1: Gather columns
        columns_df = self._get_column_selection(arg[1])
        columns_df._index = self._df._index

        # Step 2: Gather rows
        if isinstance(columns_df.index, MultiIndex):
            return columns_df.index._get_row_major(columns_df, arg[0])
        else:
            df = DataFrame()
            for col in columns_df.columns:
                # need Series() in case a scalar is returned
                df[col] = Series(columns_df[col].loc[arg[0]])
            df.columns = columns_df.columns

        # Step 3: Gather index
        if df.shape[0] == 1:  # we have a single row
            if isinstance(arg[0], slice):
                start = arg[0].start
                if start is None:
                    start = self._df.index[0]
                df.index = as_index(start)
            else:
                row_selection = column.as_column(arg[0])
                if pd.api.types.is_bool_dtype(row_selection.dtype):
                    df.index = self._df.index.take(row_selection)
                else:
                    df.index = as_index(row_selection)
        # Step 4: Downcast
        if self._can_downcast_to_series(df, arg):
            return self._downcast_to_series(df, arg)
        return df

    def _setitem_tuple_arg(self, key, value):
        if isinstance(self._df.index, cudf.MultiIndex) or isinstance(
            self._df.columns, pd.MultiIndex
        ):
            raise NotImplementedError(
                "Setting values using df.loc[] not supported on "
                "DataFrames with a MultiIndex"
            )

        columns = self._get_column_selection(key[1])

        for col in columns:
            self._df[col].loc[key[0]] = value

    def _get_column_selection(self, arg):
        return self._df._get_columns_by_label(arg)


class _DataFrameIlocIndexer(_DataFrameIndexer):
    """
    For selection by index.
    """

    def __init__(self, df):
        self._df = df

    def _getitem_tuple_arg(self, arg):
        from cudf import MultiIndex
        from cudf.core.dataframe import DataFrame, Series
        from cudf.core.index import as_index

        # Iloc Step 1:
        # Gather the columns specified by the second tuple arg
        columns_df = self._get_column_selection(arg[1])
        columns_df._index = self._df._index

        # Iloc Step 2:
        # Gather the rows specified by the first tuple arg
        if isinstance(columns_df.index, MultiIndex):
            df = columns_df.index._get_row_major(columns_df, arg[0])
            if (len(df) == 1 and len(columns_df) >= 1) and not (
                isinstance(arg[0], slice) or isinstance(arg[1], slice)
            ):
                # Pandas returns a numpy scalar in this case
                return df[0]
            if self._can_downcast_to_series(df, arg):
                return self._downcast_to_series(df, arg)
            return df
        else:
            df = DataFrame()
            for i, col in enumerate(columns_df._columns):
                # need Series() in case a scalar is returned
                df[i] = Series(col[arg[0]])

            df.index = as_index(columns_df.index[arg[0]])
            df.columns = columns_df.columns

        # Iloc Step 3:
        # Reindex
        if df.shape[0] == 1:  # we have a single row without an index
            df.index = as_index(self._df.index[arg[0]])

        # Iloc Step 4:
        # Downcast
        if self._can_downcast_to_series(df, arg):
            return self._downcast_to_series(df, arg)

        if df.shape[0] == 0 and df.shape[1] == 0 and isinstance(arg[0], slice):
            from cudf.core.index import RangeIndex

            slice_len = arg[0].stop or len(self._df)
            start, stop, step = arg[0].indices(slice_len)
            df._index = RangeIndex(start, stop)
        return df

    def _setitem_tuple_arg(self, key, value):
        columns = self._get_column_selection(key[1])

        for col in columns:
            self._df[col].iloc[key[0]] = value

    def _getitem_scalar(self, arg):
        col = self._df.columns[arg[1]]
        return self._df[col].iloc[arg[0]]

    def _get_column_selection(self, arg):
        return cudf.DataFrame(self._df._get_columns_by_index(arg))


def _normalize_dtypes(df):
    if len(df.columns) > 0:
        dtypes = df.dtypes.values.tolist()
        normalized_dtype = np.result_type(*dtypes)
        for name, col in df._data.items():
            df[name] = col.astype(normalized_dtype)
    return df
