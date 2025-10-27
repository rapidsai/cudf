# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from pylibcudf.column cimport Column
from pylibcudf.scalar cimport Scalar
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource
from rmm.pylibrmm.stream cimport Stream

ctypedef fused ColumnOrScalar:
    Column
    Scalar

cpdef Column slice_strings(
    Column input,
    ColumnOrScalar start=*,
    ColumnOrScalar stop=*,
    Scalar step=*,
    Stream stream=*,
    DeviceMemoryResource mr=*
)
