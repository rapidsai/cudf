# Copyright (c) 2024, NVIDIA CORPORATION.

from pylibcudf.column cimport Column
from pylibcudf.libcudf.datetime cimport datetime_component, rounding_frequency
from pylibcudf.scalar cimport Scalar

ctypedef fused ColumnOrScalar:
    Column
    Scalar

cpdef Column extract_millisecond_fraction(
    Column input
)

cpdef Column extract_microsecond_fraction(
    Column input
)

cpdef Column extract_nanosecond_fraction(
    Column input
)

cpdef Column extract_datetime_component(
    Column input,
    datetime_component component
)

cpdef Column ceil_datetimes(
    Column input,
    rounding_frequency freq
)

cpdef Column floor_datetimes(
    Column input,
    rounding_frequency freq
)

cpdef Column round_datetimes(
    Column input,
    rounding_frequency freq
)

cpdef Column add_calendrical_months(
    Column timestamps,
    ColumnOrScalar months,
)

cpdef Column day_of_year(Column input)

cpdef Column is_leap_year(Column input)

cpdef Column last_day_of_month(Column input)

cpdef Column extract_quarter(Column input)

cpdef Column days_in_month(Column input)
