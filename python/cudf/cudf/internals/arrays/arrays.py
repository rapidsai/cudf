from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

import cudf
from cudf import _lib as libcudf
from cudf._typing import Dtype
from cudf.utils.dtypes import (
    is_bool_dtype,
    is_categorical_dtype,
    is_datetime_dtype,
    is_decimal_dtype,
    is_float_dtype,
    is_integer_dtype,
    is_list_dtype,
    is_string_dtype,
    is_struct_dtype,
    is_timedelta_dtype,
)

if TYPE_CHECKING:
    from cudf.core.column import ColumnBase, NumericalColumn, StringColumn


class Array:
    _column: ColumnBase

    def __init__(self, column: ColumnBase):
        self._column = column

    @property
    def dtype(self):
        return self._column.dtype

    def __len__(self):
        return len(self._column)

    def _process_for_reduction(self, skipna: bool = None) -> Array:
        if skipna:
            preprocessed = asarray(self._column.nans_to_nulls())  # TODO
            preprocessed = asarray(preprocessed._column.dropna())  # TODO
        else:
            preprocessed = self

        return preprocessed

    def _reduce(self, op: str, skipna: bool = None, **kwargs):
        skipna = True if skipna is None else skipna

        if not skipna and self._column.has_nulls:
            return cudf.utils.dtypes._get_nan_for_dtype(self.dtype)

        preprocessed = self._process_for_reduction(skipna=skipna)

        min_count = kwargs.pop("min_count", 0)
        if min_count > 0:
            if preprocessed._column.valid_count < min_count:  # TODO
                return cudf.utils.dtypes._get_nan_for_dtype(self.dtype)
        elif min_count < 0:
            warnings.warn(
                f"min_count value cannot be negative({min_count}), will "
                f"default to 0."
            )

        return libcudf.reduce.reduce(op, preprocessed._column, **kwargs)

    def astype(self, dtype: Dtype, **kwargs) -> Array:
        return asarray(self._column.astype(dtype, **kwargs))

    def max(self, skipna: bool = None, dtype: Dtype = None) -> float:
        return self._reduce("max", skipna=skipna, dtype=dtype)

    def min(self, skipna: bool = None, dtype: Dtype = None) -> float:
        return self._reduce("min", skipna=skipna, dtype=dtype)

    def sum(
        self, skipna: bool = None, dtype: Dtype = None, min_count: int = 0
    ):
        raise TypeError(f"cannot perform sum with type {self.dtype}")

    def product(
        self, skipna: bool = None, dtype: Dtype = None, min_count: int = 0
    ):
        raise TypeError(f"cannot perform product with type {self.dtype}")

    def mean(self, skipna: bool = None, dtype: Dtype = None):
        raise TypeError(f"cannot perform mean with type {self.dtype}")

    def std(self, skipna: bool = None, ddof: int = 1, dtype: Dtype = None):
        raise TypeError(f"cannot perform std with type {self.dtype}")

    def var(
        self, skipna: bool = None, ddof: int = 1, dtype: Dtype = None
    ) -> float:
        raise TypeError(f"cannot perform var with type {self.dtype}")

    def kurtosis(self, skipna: bool = None):
        raise TypeError(f"cannot perform kurtosis with type {self.dtype}")

    def skew(self, skipna: bool = None):
        raise TypeError(f"cannot perform skew with type {self.dtype}")

    def view(self, dtype):
        return asarray(self._column.view(dtype))


class _NumericArray(Array):
    _column: NumericalColumn

    def product(
        self, skipna: bool = None, dtype: Dtype = None, min_count: int = 0
    ) -> float:
        return self._reduce(
            "product", skipna=skipna, dtype=dtype, min_count=min_count
        )

    def mean(self, skipna: bool = None, dtype: Dtype = None) -> float:
        dtype = np.float64 if dtype is None else dtype
        return self._reduce("mean", skipna=skipna, dtype=dtype)

    def median(self, skipna: bool = None, dtype: Dtype = None) -> float:
        dtype = np.float64 if dtype is None else dtype
        return self._reduce("median", skipna=skipna, dtype=dtype)

    def var(
        self, skipna: bool = None, ddof: int = 1, dtype: Dtype = None
    ) -> float:
        dtype = np.float64 if dtype is None else dtype
        return self._reduce("var", skipna=skipna, dtype=dtype, ddof=ddof)

    def std(
        self, skipna: bool = None, ddof: int = 1, dtype: Dtype = None
    ) -> float:
        dtype = np.float64 if dtype is None else dtype
        return self._reduce("std", skipna=skipna, dtype=dtype, ddof=ddof)

    def sum(
        self, skipna: bool = None, dtype: Dtype = None, min_count: int = 0
    ) -> float:
        return self._reduce(
            "sum", skipna=skipna, dtype=dtype, min_count=min_count
        )

    def sum_of_squares(
        self, skipna: bool = None, dtype: Dtype = None
    ) -> float:
        return self._reduce("sum_of_squares", dtype=dtype)

    def kurtosis(self, skipna: bool = None) -> float:
        return self._column.kurtosis(skipna=skipna)

    def skew(self, skipna: bool = None) -> float:
        return self._column.skew(skipna=skipna)


class IntegerArray(_NumericArray):
    pass


class BooleanArray(_NumericArray):
    pass


class FloatingArray(_NumericArray):
    pass


class StringArray(Array):
    _column: StringColumn

    def sum(
        self, skipna: bool = None, dtype: Dtype = None, min_count: int = 0
    ):
        return self._column.sum(
            skipna=skipna, dtype=dtype, min_count=min_count
        )


class CategoricalArray(Array):
    pass


class DatetimeArray(Array):

    _numpy_to_pandas_conversion = {
        "ns": 1,
        "us": 1000,
        "ms": 1000000,
        "s": 1000000000,
        "m": 60000000000,
        "h": 3600000000000,
        "D": 86400000000000,
    }

    @property
    def _time_unit(self):
        return np.datetime_data(self.dtype)[0]

    def mean(self, skipna: bool = None, dtype: Dtype = None):
        return pd.Timestamp(
            self.view("int64").mean(skipna=skipna, dtype="float64"),
            unit=self._time_unit,
        )

    def std(self, skipna: bool = None, ddof: int = 1, dtype: Dtype = None):
        return pd.Timedelta(
            self.view("int64").std(skipna=skipna, ddof=ddof, dtype="float64")
            * self._numpy_to_pandas_conversion[self._time_unit],
        )

    def median(self, skipna: bool = None) -> pd.Timestamp:
        return pd.Timestamp(
            self.view("int64").median(skipna=skipna), unit=self._time_unit
        )


class TimedeltaArray(Array):

    _numpy_to_pandas_conversion = {
        "ns": 1,
        "us": 1000,
        "ms": 1000000,
        "s": 1000000000,
        "m": 60000000000,
        "h": 3600000000000,
        "D": 86400000000000,
    }

    @property
    def _time_unit(self):
        return np.datetime_data(self.dtype)[0]

    def mean(self, skipna: bool = None, dtype: Dtype = None):
        return pd.Timedelta(
            self.view("int64").mean(skipna=skipna, dtype="float64"),
            unit=self._time_unit,
        )

    def median(self, skipna: bool = None) -> pd.Timestamp:
        return pd.Timedelta(
            self.view("int64").median(skipna=skipna), unit=self._time_unit
        )

    def sum(
        self, skipna: bool = None, dtype: Dtype = None, min_count=0
    ) -> pd.Timedelta:
        return pd.Timedelta(
            self.view("int64").sum(
                skipna=skipna, dtype=dtype, min_count=min_count
            ),
            unit=self._time_unit,
        )

    def std(
        self, skipna: bool = None, ddof: int = 1, dtype: Dtype = None
    ) -> pd.Timedelta:
        return pd.Timedelta(
            self.view("int64").std(skipna=skipna, ddof=ddof, dtype="float64"),
            unit=self._time_unit,
        )


class DecimalArray(_NumericArray):
    pass


class ListArray(Array):
    pass


class StructArray(Array):
    pass


def asarray(data, dtype=None) -> Array:
    from cudf.core.column import as_column

    column = as_column(data, dtype)

    if is_categorical_dtype(column.dtype):
        return CategoricalArray(column)
    elif is_list_dtype(column.dtype):
        return ListArray(column)
    elif is_struct_dtype(column.dtype):
        return StructArray(column)
    elif is_integer_dtype(column.dtype):
        return IntegerArray(column)
    elif is_bool_dtype(column.dtype):
        return BooleanArray(column)
    elif is_float_dtype(column.dtype):
        return FloatingArray(column)
    elif is_string_dtype(column.dtype):
        return StringArray(column)
    elif is_datetime_dtype(column.dtype):
        return DatetimeArray(column)
    elif is_timedelta_dtype(column.dtype):
        return TimedeltaArray(column)
    elif is_decimal_dtype(column.dtype):
        return DecimalArray(column)
    else:
        raise TypeError(f"Unrecognized dtype {dtype}")
