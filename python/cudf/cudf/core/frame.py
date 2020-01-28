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

    def gather(self, gather_map):
        if not pd.api.types.is_integer_dtype(gather_map.dtype):
            gather_map = gather_map.astype("int32")
        result = self._from_table(
            libcudfxx.gather(self, as_column(gather_map))
        )
        result._copy_categories(self)
        return result

    def drop_nulls(self, how="any", keys=None, thresh=None):
        """
        Drops null rows from `self` depending on key columns.
        """
        result = self._from_table(
            libcudfxx.drop_nulls(self, how, keys, thresh)
        )
        result._copy_categories(self)
        return result

    def apply_boolean_mask(self, boolean_mask):
        """
        Applies boolean mask to each row of `self`, rows corresponding to `False`
        is dropped
        """
        result = self._from_table(
            libcudfxx.apply_boolean_mask(self, as_column(boolean_mask))
        )
        result._copy_categories(self)
        return result

    def frame_drop_duplicates(self, keys=None, keep='first', nulls_are_equal=True):
        """
        Drops rows in frame as per duplicate rows in `keys` columns.
        """
        result = self._from_table(
            libcudfxx.drop_duplicates(self, keys, keep, nulls_are_equal)
        )

        result._copy_categories(self)
        return result

    def unique_count(self, ignore_nulls=True, nan_as_null=False):
        """
        Finds number of unique rows in `source_table`
        """

        return libcudfxx.unique_count(self, ignore_nulls, nan_as_null)

    def _copy_categories(self, other, include_index=True):
        """
        Utility that copies category information from `other`
        to `self`.
        """
        for name, col in other._data.items():
            if is_categorical_dtype(col.dtype):
                self._data[name] = build_categorical_column(
                    categories=col.categories,
                    codes=self._data[name],
                    mask=self._data[name].mask,
                    ordered=col.ordered,
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
