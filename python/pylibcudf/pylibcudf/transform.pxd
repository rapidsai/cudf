# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from libcpp cimport bool
from pylibcudf.libcudf.types cimport bitmask_type, data_type, null_aware
from rmm.pylibrmm.stream cimport Stream
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource

from .column cimport Column
from .expressions cimport Expression
from .gpumemoryview cimport gpumemoryview
from .table cimport Table
from .types cimport DataType


cpdef tuple[gpumemoryview, int] nans_to_nulls(
    Column input, Stream stream = *, DeviceMemoryResource mr = *
)

cpdef Column compute_column(
    Table input, Expression expr, Stream stream = *, DeviceMemoryResource mr = *
)

cpdef tuple[gpumemoryview, int] bools_to_mask(
    Column input, Stream stream = *, DeviceMemoryResource mr = *
)

cpdef Column mask_to_bools(
    Py_ssize_t bitmask,
    int begin_bit,
    int end_bit,
    Stream stream = *,
    DeviceMemoryResource mr = *,
)

cpdef Column transform(
    list[Column] inputs,
    str transform_udf,
    DataType output_type,
    bool is_ptx,
    null_aware is_null_aware,
    Stream stream = *,
    DeviceMemoryResource mr = *,
)

cpdef tuple[Table, Column] encode(
    Table input, Stream stream = *, DeviceMemoryResource mr = *
)

cpdef Table one_hot_encode(
    Column input_column,
    Column categories,
    Stream stream = *,
    DeviceMemoryResource mr = *,
)
