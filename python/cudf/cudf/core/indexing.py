# Copyright (c) 2020, NVIDIA CORPORATION.
import cupy
import numpy as np
import pandas as pd

import cudf
from cudf._lib.nvtx import annotate
from cudf.utils.dtypes import (
    is_categorical_dtype,
    is_column_like,
    is_list_like,
    is_scalar,
    to_cudf_compatible_scalar,
)


def indices_from_labels(obj, labels):
    from cudf.core.column import column

    if not isinstance(labels, cudf.MultiIndex):
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

    # join is not guaranteed to maintain the index ordering
    # so we will sort it with its initial ordering which is stored
    # in column "__"
    lhs = cudf.DataFrame({"__": cupy.arange(len(labels))}, index=labels)
    rhs = cudf.DataFrame({"_": cupy.arange(len(obj))}, index=obj.index)
    return lhs.join(rhs).sort_values("__")["_"]


def get_label_range_or_mask(index, start, stop, step):
    if (
        not (start is None and stop is None)
        and type(index) is cudf.core.index.DatetimeIndex
        and index.is_monotonic is False
    ):
        start = pd.to_datetime(start)
        stop = pd.to_datetime(stop)
        if start is not None and stop is not None:
            if start > stop:
                return slice(0, 0, None)
            boolean_mask = (index >= start) and (index <= stop)
        elif start is not None:
            boolean_mask = index >= start
        else:
            boolean_mask = index <= stop
        return boolean_mask
    else:
        start, stop = index.find_label_range(start, stop)
        return slice(start, stop, step)


class _SeriesIlocIndexer(object):
    """
    For integer-location based selection.
    """

    def __init__(self, sr):
        self._sr = sr

    def __getitem__(self, arg):
        if isinstance(arg, tuple):
            arg = list(arg)
        data = self._sr._column[arg]
        if is_scalar(data) or data is None:
            return data
        index = self._sr.index.take(arg)
        return self._sr._copy_construct(data=data, index=index)

    def __setitem__(self, key, value):
        from cudf.core.column import column

        if isinstance(key, tuple):
            key = list(key)

        # coerce value into a scalar or column
        if is_scalar(value):
            value = to_cudf_compatible_scalar(value)
        else:
            value = column.as_column(value)

        if hasattr(value, "dtype") and pd.api.types.is_numeric_dtype(
            value.dtype
        ):
            # normalize types if necessary:
            if not pd.api.types.is_integer(key):
                to_dtype = np.result_type(value.dtype, self._sr._column.dtype)
                value = value.astype(to_dtype)
                self._sr._column._mimic_inplace(
                    self._sr._column.astype(to_dtype), inplace=True
                )

        self._sr._column[key] = value


class _SeriesLocIndexer(object):
    """
    Label-based selection
    """

    def __init__(self, sr):
        self._sr = sr

    def __getitem__(self, arg):
        try:
            arg = self._loc_to_iloc(arg)
        except (TypeError, KeyError, IndexError, ValueError):
            raise IndexError("Failed to convert index to appropirate row")

        return self._sr.iloc[arg]

    def __setitem__(self, key, value):
        key = self._loc_to_iloc(key)
        if isinstance(value, (pd.Series, cudf.Series)):
            value = cudf.Series(value)
            value = value._align_to_index(self._sr.index, how="right")
        self._sr.iloc[key] = value

    def _loc_to_iloc(self, arg):
        from cudf.core.column import column
        from cudf.core.series import Series

        if is_scalar(arg):
            try:
                found_index = self._sr.index._values.find_first_value(
                    arg, closest=False
                )
                return found_index
            except (TypeError, KeyError, IndexError, ValueError):
                raise IndexError("label scalar is out of bound")

        elif isinstance(arg, slice):
            return get_label_range_or_mask(
                self._sr.index, arg.start, arg.stop, arg.step
            )
        elif isinstance(arg, (cudf.MultiIndex, pd.MultiIndex)):
            if isinstance(arg, pd.MultiIndex):
                arg = cudf.MultiIndex.from_pandas(arg)

            return indices_from_labels(self._sr, arg)

        else:
            arg = Series(column.as_column(arg))
            if arg.dtype in [np.bool, np.bool_]:
                return arg
            else:
                indices = indices_from_labels(self._sr, arg)
                if indices.null_count > 0:
                    raise IndexError("label scalar is out of bound")
                return indices


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
            elif (is_list_like(arg[0]) or is_column_like(arg[0])) and (
                is_list_like(arg[1])
                or is_column_like(arg[0])
                or type(arg[1]) is slice
            ):
                return False
            else:
                if pd.api.types.is_bool_dtype(
                    as_column(arg[0]).dtype
                ) and not isinstance(arg[1], slice):
                    return True
            dtypes = df.dtypes.values.tolist()
            all_numeric = all(
                [pd.api.types.is_numeric_dtype(t) for t in dtypes]
            )
            if all_numeric:
                return True
        if ncols == 1:
            if type(arg[1]) is slice:
                return False
            if isinstance(arg[1], tuple):
                # Multiindex indexing with a slice
                if any(isinstance(v, slice) for v in arg):
                    return False
            if not (is_list_like(arg[1]) or is_column_like(arg[1])):
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
                return df[df.columns[0]].iloc[0]
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

    @annotate("LOC_GETITEM", color="blue", domain="cudf_python")
    def _getitem_tuple_arg(self, arg):
        from uuid import uuid4

        from cudf import MultiIndex
        from cudf.core.column import column
        from cudf.core.dataframe import DataFrame
        from cudf.core.index import as_index

        # Step 1: Gather columns
        if isinstance(arg, tuple):
            columns_df = self._get_column_selection(arg[1])
            columns_df._index = self._df._index
        else:
            columns_df = self._df

        # Step 2: Gather rows
        if isinstance(columns_df.index, MultiIndex):
            if isinstance(arg, (MultiIndex, pd.MultiIndex)):
                if isinstance(arg, pd.MultiIndex):
                    arg = MultiIndex.from_pandas(arg)

                indices = indices_from_labels(columns_df, arg)
                return columns_df.take(indices)

            else:
                return columns_df.index._get_row_major(columns_df, arg[0])
        else:
            if isinstance(arg[0], slice):
                out = get_label_range_or_mask(
                    columns_df.index, arg[0].start, arg[0].stop, arg[0].step
                )
                if isinstance(out, slice):
                    df = columns_df._slice(out)
                else:
                    df = columns_df._apply_boolean_mask(out)
            else:
                tmp_arg = arg
                if is_scalar(arg[0]):
                    # If a scalar, there is possibility of having duplicates.
                    # Join would get all the duplicates. So, coverting it to
                    # an array kind.
                    tmp_arg = ([tmp_arg[0]], tmp_arg[1])
                if len(tmp_arg[0]) == 0:
                    return columns_df._empty_like(keep_index=True)
                tmp_arg = (column.as_column(tmp_arg[0]), tmp_arg[1])

                if pd.api.types.is_bool_dtype(tmp_arg[0]):
                    df = columns_df._apply_boolean_mask(tmp_arg[0])
                else:
                    tmp_col_name = str(uuid4())
                    other_df = DataFrame(
                        {tmp_col_name: cupy.arange(len(tmp_arg[0]))},
                        index=as_index(tmp_arg[0]),
                    )
                    df = other_df.join(columns_df, how="inner")
                    # as join is not assigning any names to index,
                    # update it over here
                    df.index.name = columns_df.index.name
                    df = df.sort_values(tmp_col_name)
                    df.drop([tmp_col_name], inplace=True)
                    # There were no indices found
                    if len(df) == 0:
                        raise IndexError

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

    @annotate("LOC_SETITEM", color="blue", domain="cudf_python")
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

    @annotate("ILOC_GETITEM", color="blue", domain="cudf_python")
    def _getitem_tuple_arg(self, arg):
        from cudf import MultiIndex
        from cudf.core.column import column
        from cudf.core.index import as_index

        # Iloc Step 1:
        # Gather the columns specified by the second tuple arg
        columns_df = self._get_column_selection(arg[1])
        columns_df._index = self._df._index

        # Iloc Step 2:
        # Gather the rows specified by the first tuple arg
        if isinstance(columns_df.index, MultiIndex):
            if isinstance(arg[0], slice):
                df = columns_df[arg[0]]
            else:
                df = columns_df.index._get_row_major(columns_df, arg[0])
            if (len(df) == 1 and len(columns_df) >= 1) and not (
                isinstance(arg[0], slice) or isinstance(arg[1], slice)
            ):
                # Pandas returns a numpy scalar in this case
                return df.iloc[0]
            if self._can_downcast_to_series(df, arg):
                return self._downcast_to_series(df, arg)
            return df
        else:
            if isinstance(arg[0], slice):
                df = columns_df._slice(arg[0])
            elif is_scalar(arg[0]):
                index = arg[0]
                if index < 0:
                    index += len(columns_df)
                df = columns_df._slice(slice(index, index + 1, 1))
            else:
                arg = (column.as_column(arg[0]), arg[1])
                if pd.api.types.is_bool_dtype(arg[0]):
                    df = columns_df._apply_boolean_mask(arg[0])
                else:
                    df = columns_df._gather(arg[0])

        # Iloc Step 3:
        # Reindex
        if df.shape[0] == 1:  # we have a single row without an index
            df.index = as_index(self._df.index[arg[0]])

        # Iloc Step 4:
        # Downcast
        if self._can_downcast_to_series(df, arg):
            return self._downcast_to_series(df, arg)

        if df.shape[0] == 0 and df.shape[1] == 0 and isinstance(arg[0], slice):
            df._index = as_index(self._df.index[arg[0]])
        return df

    @annotate("ILOC_SETITEM", color="blue", domain="cudf_python")
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
