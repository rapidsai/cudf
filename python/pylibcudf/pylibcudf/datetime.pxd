# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from pylibcudf.column cimport Column
from pylibcudf.libcudf.datetime cimport datetime_component, rounding_frequency
from pylibcudf.scalar cimport Scalar
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource
from rmm.pylibrmm.stream cimport Stream

ctypedef fused ColumnOrScalar:
    Column
    Scalar

cpdef Column extract_datetime_component(
    Column input,
    datetime_component component,
    Stream stream = *,
    DeviceMemoryResource mr = *,
)

cpdef Column ceil_datetimes(
    Column input,
    rounding_frequency freq,
    Stream stream = *,
    DeviceMemoryResource mr = *,
)

cpdef Column floor_datetimes(
    Column input,
    rounding_frequency freq,
    Stream stream = *,
    DeviceMemoryResource mr = *,
)

cpdef Column round_datetimes(
    Column input,
    rounding_frequency freq,
    Stream stream = *,
    DeviceMemoryResource mr = *,
)

cpdef Column add_calendrical_months(
    Column timestamps,
    ColumnOrScalar months,
    Stream stream = *,
    DeviceMemoryResource mr = *,
)

cpdef Column day_of_year(
    Column input, Stream stream = *, DeviceMemoryResource mr = *
)

cpdef Column is_leap_year(
    Column input, Stream stream = *, DeviceMemoryResource mr = *
)

cpdef Column last_day_of_month(
    Column input, Stream stream = *, DeviceMemoryResource mr = *
)

cpdef Column extract_quarter(
    Column input, Stream stream = *, DeviceMemoryResource mr = *
)

cpdef Column days_in_month(
    Column input, Stream stream = *, DeviceMemoryResource mr = *
)
