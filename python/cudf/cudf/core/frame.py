import numpy as np
import pandas as pd

import cudf._libxx as libcudfxx
from cudf.core.column import as_column, build_categorical_column
from cudf.utils.dtypes import is_categorical_dtype


class Frame(libcudfxx.Table):
    """
    Frame: A collection of Column objects with an optional index.

    Parameters
    ----------
    data : OrderedColumnDict
        An OrderedColumnDict mapping column names to Columns
    index : Table
        A Frame representing the (optional) index columns.
    """

    @classmethod
    def _from_table(cls, table):
        return cls(table._data, index=table._index)

    def _gather(self, gather_map):
        if not pd.api.types.is_integer_dtype(gather_map.dtype):
            gather_map = gather_map.astype("int32")
        result = self.__class__._from_table(
            libcudfxx.gather(self, as_column(gather_map))
        )
        result._copy_categories(self)
        return result

    def _hash_partition(self, columns_to_hash, num_partitions):
        frame, offsets = libcudfxx._hash_partition(
            self, columns_to_hash, num_partitions)
        frame._copy_categories(self)
        return frame, offsets

    def _hash_columns(self):
        result = libcudfxx._hash(self)
        return result

    def _as_column(self):
        """
        _as_column : Converts a single columned Frame to Column
        """
        assert (
            self._num_columns == 1
            and self._index is None
            and self._column_names[0] is None
        ), """There should be only one data column,
            no index and None as the name to use this method"""

        return self._data[None].copy(deep=False)

    def dropna(self, axis=0, how="any", subset=None, thresh=None):
        """
        Drops rows (or columns) containing nulls from a Column.

        Parameters
        ----------
        axis : {0, 1}, optional
            Whether to drop rows (axis=0, default) or columns (axis=1)
            containing nulls.
        how : {"any", "all"}, optional
            Specifies how to decide whether to drop a row (or column).
            any (default) drops rows (or columns) containing at least
            one null value. all drops only rows (or columns) containing
            *all* null values.
        subset : list, optional
            List of columns to consider when dropping rows (all columns
            are considered by default). Alternatively, when dropping
            columns, subset is a list of rows to consider.
        thresh: int, optional
            If specified, then drops every row (or column) containing
            less than `thresh` non-null values


        Returns
        -------
        Copy of the DataFrame with rows/columns containing nulls dropped.
        """
        if axis == 0:
            return self._drop_na_rows(how=how, subset=subset, thresh=thresh)
        else:
            return self._drop_na_columns(how=how, subset=subset, thresh=thresh)

    def _drop_na_rows(self, how="any", subset=None, thresh=None):
        """
        Drops null rows from `self`.

        how : {"any", "all"}, optional
            Specifies how to decide whether to drop a row.
            any (default) drops rows containing at least
            one null value. all drops only rows containing
            *all* null values.
        subset : list, optional
            List of columns to consider when dropping rows.
        thresh: int, optional
            If specified, then drops every row containing
            less than `thresh` non-null values.
        """
        if subset is None:
            subset = self._column_names
        elif (
            not np.iterable(subset)
            or isinstance(subset, str)
            or isinstance(subset, tuple)
            and subset in self.columns
        ):
            subset = (subset,)
        diff = set(subset) - set(self._data)
        if len(diff) != 0:
            raise KeyError("columns {!r} do not exist".format(diff))
        subset_cols = [
            name for name, col in self._data.items() if name in subset
        ]
        if len(subset_cols) == 0:
            return self.copy(deep=True)
        result = self.__class__._from_table(
            libcudfxx.drop_nulls(self, how=how, keys=subset, thresh=thresh)
        )
        result._copy_categories(self)
        return result

    def _drop_na_columns(self, how="any", subset=None, thresh=None):
        """
        Drop columns containing nulls
        """
        out_cols = []

        if subset is None:
            df = self
        else:
            df = self.take(subset)

        if thresh is None:
            if how == "all":
                thresh = 1
            else:
                thresh = len(df)

        for col in self.columns:
            if (len(df[col]) - df[col].null_count) < thresh:
                continue
            out_cols.append(col)

        return self[out_cols]

    def _apply_boolean_mask(self, boolean_mask):
        """
        Applies boolean mask to each row of `self`,
        rows corresponding to `False` is dropped
        """
        result = self._from_table(
            libcudfxx.apply_boolean_mask(self, as_column(boolean_mask))
        )
        result._copy_categories(self)
        return result

    def drop_duplicates(self, subset=None, keep="first", nulls_are_equal=True):
        """
        Drops rows in frame as per duplicate rows in `seubset` columns from
        self.

        subset : list, optional
            List of columns to consider when dropping rows.
        keep : ["first", "last", False] first will keep first of duplicate,
            last will keep last of the duplicate and False drop all
            duplicate
        nulls_are_equal: null elements are considered equal to other null
            elements
        """
        if subset is None:
            subset = self._column_names
        elif (
            not np.iterable(subset)
            or isinstance(subset, str)
            or isinstance(subset, tuple)
            and subset in self.columns
        ):
            subset = (subset,)
        diff = set(subset) - set(self._data)
        if len(diff) != 0:
            raise KeyError("columns {!r} do not exist".format(diff))
        subset_cols = [name for name in self._column_names if name in subset]
        if len(subset_cols) == 0:
            return self.copy(deep=True)

        result = self._from_table(
            libcudfxx.drop_duplicates(
                self, keys=subset, keep=keep, nulls_are_equal=nulls_are_equal
            )
        )

        result._copy_categories(self)
        return result

    def _copy_categories(self, other, include_index=True):
        """
        Utility that copies category information from `other`
        to `self`.
        """
        for name, col, other_col in zip(
            self._column_names, self._columns, other._columns
        ):
            if is_categorical_dtype(other_col) and not is_categorical_dtype(
                col
            ):
                self._data[name] = build_categorical_column(
                    categories=other_col.categories,
                    codes=col,
                    mask=col.mask,
                    ordered=other_col.ordered,
                )
        if include_index:
            if self._index is not None:
                self._index._copy_categories(other._index)
        return self

    def _unaryop(self, op):
        result = self.copy()
        for name, col in result._data.items():
            result._data[name] = col.unary_operator(op)
        return result

    def sin(self):
        return self._unaryop("sin")

    def cos(self):
        return self._unaryop("cos")

    def tan(self):
        return self._unaryop("tan")

    def asin(self):
        return self._unaryop("asin")

    def acos(self):
        return self._unaryop("acos")

    def atan(self):
        return self._unaryop("atan")

    def exp(self):
        return self._unaryop("exp")

    def log(self):
        return self._unaryop("log")

    def sqrt(self):
        return self._unaryop("sqrt")
