from __future__ import annotations

from typing import TYPE_CHECKING

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
    from cudf.core.column import ColumnBase


class Array:
    _column: ColumnBase

    def __init__(self, column: ColumnBase):
        self._column = column

    @property
    def dtype(self):
        return self._column.dtype


class IntegerArray(Array):
    pass


class BooleanArray(Array):
    pass


class FloatingArray(Array):
    pass


class StringArray(Array):
    pass


class CategoricalArray(Array):
    pass


class DatetimeArray(Array):
    pass


class TimedeltaArray(Array):
    pass


class DecimalArray(Array):
    pass


class ListArray(Array):
    pass


class StructArray(Array):
    pass


def array(data, dtype=None):
    from cudf.core.column import as_column

    column = as_column(data, dtype)

    if is_integer_dtype(column.dtype):
        return IntegerArray(column)
    elif is_bool_dtype(column.dtype):
        return BooleanArray(column)
    elif is_float_dtype(column.dtype):
        return FloatingArray(column)
    elif is_string_dtype(column.dtype):
        return StringArray(column)
    elif is_categorical_dtype(column.dtype):
        return CategoricalArray(column)
    elif is_datetime_dtype(column.dtype):
        return DatetimeArray(column)
    elif is_timedelta_dtype(column.dtype):
        return TimedeltaArray(column)
    elif is_decimal_dtype(column.dtype):
        return DecimalArray(column)
    elif is_list_dtype(column.dtype):
        return ListArray(column)
    elif is_struct_dtype(column.dtype):
        return StructArray(column)
    else:
        raise TypeError(f"Unrecognized dtype {dtype}")
