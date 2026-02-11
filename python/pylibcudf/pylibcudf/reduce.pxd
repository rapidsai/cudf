# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from libcpp cimport bool
from pylibcudf.libcudf.reduce cimport scan_type
from pylibcudf.libcudf.types cimport nan_policy, null_policy, size_type
from rmm.pylibrmm.stream cimport Stream
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource

from .aggregation cimport Aggregation
from .column cimport Column
from .scalar cimport Scalar
from .types cimport DataType


cpdef Scalar reduce(
    Column col,
    Aggregation agg,
    DataType data_type,
    Scalar init = *,
    Stream stream = *,
    DeviceMemoryResource mr = *,
)

cpdef Column scan(
    Column col,
    Aggregation agg,
    scan_type inclusive,
    Stream stream = *,
    DeviceMemoryResource mr = *,
)

cpdef tuple minmax(Column col, Stream stream = *, DeviceMemoryResource mr = *)

cpdef bool is_valid_reduce_aggregation(DataType source, Aggregation agg)

cpdef size_type unique_count(
    Column source,
    null_policy null_handling,
    nan_policy nan_handling,
    Stream stream = *
)

cpdef size_type distinct_count(
    Column source,
    null_policy null_handling,
    nan_policy nan_handling,
    Stream stream = *
)
