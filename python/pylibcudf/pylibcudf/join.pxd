# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from pylibcudf.libcudf.types cimport null_equality
from rmm.pylibrmm.stream cimport Stream
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource

from .column cimport Column
from .expressions cimport Expression
from .table cimport Table


cpdef tuple inner_join(
    Table left_keys,
    Table right_keys,
    null_equality nulls_equal,
    Stream stream=*,
    DeviceMemoryResource mr=*,
)

cpdef tuple left_join(
    Table left_keys,
    Table right_keys,
    null_equality nulls_equal,
    Stream stream=*,
    DeviceMemoryResource mr=*,
)

cpdef tuple full_join(
    Table left_keys,
    Table right_keys,
    null_equality nulls_equal,
    Stream stream=*,
    DeviceMemoryResource mr=*,
)

cpdef Column left_semi_join(
    Table left_keys,
    Table right_keys,
    null_equality nulls_equal,
    Stream stream=*,
    DeviceMemoryResource mr=*,
)

cpdef Column left_anti_join(
    Table left_keys,
    Table right_keys,
    null_equality nulls_equal,
    Stream stream=*,
    DeviceMemoryResource mr=*,
)

cpdef Table cross_join(
    Table left, Table right, Stream stream=*, DeviceMemoryResource mr=*
)

cpdef tuple conditional_inner_join(
    Table left,
    Table right,
    Expression binary_predicate,
    Stream stream=*,
    DeviceMemoryResource mr=*,
)

cpdef tuple conditional_left_join(
    Table left,
    Table right,
    Expression binary_predicate,
    Stream stream=*,
    DeviceMemoryResource mr=*,
)

cpdef tuple conditional_full_join(
    Table left,
    Table right,
    Expression binary_predicate,
    Stream stream=*,
    DeviceMemoryResource mr=*,
)

cpdef Column conditional_left_semi_join(
    Table left,
    Table right,
    Expression binary_predicate,
    Stream stream=*,
    DeviceMemoryResource mr=*,
)

cpdef Column conditional_left_anti_join(
    Table left,
    Table right,
    Expression binary_predicate,
    Stream stream=*,
    DeviceMemoryResource mr=*,
)

cpdef tuple mixed_inner_join(
    Table left_keys,
    Table right_keys,
    Table left_conditional,
    Table right_conditional,
    Expression binary_predicate,
    null_equality nulls_equal,
    Stream stream=*,
    DeviceMemoryResource mr=*,
)

cpdef tuple mixed_left_join(
    Table left_keys,
    Table right_keys,
    Table left_conditional,
    Table right_conditional,
    Expression binary_predicate,
    null_equality nulls_equal,
    Stream stream=*,
    DeviceMemoryResource mr=*,
)

cpdef tuple mixed_full_join(
    Table left_keys,
    Table right_keys,
    Table left_conditional,
    Table right_conditional,
    Expression binary_predicate,
    null_equality nulls_equal,
    Stream stream=*,
    DeviceMemoryResource mr=*,
)

cpdef Column mixed_left_semi_join(
    Table left_keys,
    Table right_keys,
    Table left_conditional,
    Table right_conditional,
    Expression binary_predicate,
    null_equality nulls_equal,
    Stream stream=*,
    DeviceMemoryResource mr=*,
)

cpdef Column mixed_left_anti_join(
    Table left_keys,
    Table right_keys,
    Table left_conditional,
    Table right_conditional,
    Expression binary_predicate,
    null_equality nulls_equal,
    Stream stream=*,
    DeviceMemoryResource mr=*,
)
