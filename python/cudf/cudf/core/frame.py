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
        result._update_index_name(self)
        result._copy_categories(self)
        return result

    def dropna(self, how="any", subset=None, thresh=None):
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
            subset = self._data
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
        result._update_index_name(self)
        result._copy_categories(self)
        return result

    def _frame_apply_boolean_mask(self, boolean_mask):
        """
        Applies boolean mask to each row of `self`,
        rows corresponding to `False` is dropped
        """
        result = self._from_table(
            libcudfxx.apply_boolean_mask(self, as_column(boolean_mask))
        )
        result._update_index_name(self)
        result._copy_categories(self)
        return result

    def _drop_duplicates(self, keys=None, keep="first", nulls_are_equal=True):
        """
        Drops rows in frame as per duplicate rows in `keys` columns.
        """
        result = self._from_table(
            libcudfxx.drop_duplicates(self, keys, keep, nulls_are_equal)
        )

        result._update_index_name(self)
        result._copy_categories(self)
        return result

    def _update_index_name(self, other):
        """
        Update index names
        """
        if hasattr(other._index, "names"):
            if other._index.names is not None:
                self._index.names = other._index.names.copy()
            else:
                self._index.names = None

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
