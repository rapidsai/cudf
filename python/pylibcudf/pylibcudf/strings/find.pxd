# Copyright (c) 2024-2025, NVIDIA CORPORATION.

from pylibcudf.column cimport Column
from pylibcudf.libcudf.types cimport size_type
from pylibcudf.scalar cimport Scalar
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource
from rmm.pylibrmm.stream cimport Stream

ctypedef fused ColumnOrScalar:
    Column
    Scalar

cpdef Column find(
    Column input,
    ColumnOrScalar target,
    size_type start=*,
    size_type stop=*,
    Stream stream=*,
    DeviceMemoryResource mr=*,
)

cpdef Column rfind(
    Column input,
    Scalar target,
    size_type start=*,
    size_type stop=*,
    Stream stream=*,
    DeviceMemoryResource mr=*,
)

cpdef Column contains(
    Column input,
    ColumnOrScalar target,
    Stream stream=*,
    DeviceMemoryResource mr=*,
)

cpdef Column starts_with(
    Column input,
    ColumnOrScalar target,
    Stream stream=*,
    DeviceMemoryResource mr=*,
)

cpdef Column ends_with(
    Column input,
    ColumnOrScalar target,
    Stream stream=*,
    DeviceMemoryResource mr=*,
)
