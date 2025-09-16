# Copyright (c) 2024-2025, NVIDIA CORPORATION.

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
    Stream stream=*
)

cpdef tuple left_join(
    Table left_keys,
    Table right_keys,
    null_equality nulls_equal,
    Stream stream=*
)

cpdef tuple full_join(
    Table left_keys,
    Table right_keys,
    null_equality nulls_equal,
    Stream stream=*
)

cpdef Column left_semi_join(
    Table left_keys,
    Table right_keys,
    null_equality nulls_equal,
    Stream stream=*
)

cpdef Column left_anti_join(
    Table left_keys,
    Table right_keys,
    null_equality nulls_equal,
    Stream stream=*
)

cpdef Table cross_join(
    Table left, Table right, Stream stream=*, DeviceMemoryResource mr=*
)

cpdef tuple conditional_inner_join(
    Table left,
    Table right,
    Expression binary_predicate,
    Stream stream=*
)

cpdef tuple conditional_left_join(
    Table left,
    Table right,
    Expression binary_predicate,
    Stream stream=*
)

cpdef tuple conditional_full_join(
    Table left,
    Table right,
    Expression binary_predicate,
    Stream stream=*
)

cpdef Column conditional_left_semi_join(
    Table left,
    Table right,
    Expression binary_predicate,
    Stream stream=*
)

cpdef Column conditional_left_anti_join(
    Table left,
    Table right,
    Expression binary_predicate,
    Stream stream=*
)

cpdef tuple mixed_inner_join(
    Table left_keys,
    Table right_keys,
    Table left_conditional,
    Table right_conditional,
    Expression binary_predicate,
    null_equality nulls_equal,
    Stream stream=*
)

cpdef tuple mixed_left_join(
    Table left_keys,
    Table right_keys,
    Table left_conditional,
    Table right_conditional,
    Expression binary_predicate,
    null_equality nulls_equal,
    Stream stream=*
)

cpdef tuple mixed_full_join(
    Table left_keys,
    Table right_keys,
    Table left_conditional,
    Table right_conditional,
    Expression binary_predicate,
    null_equality nulls_equal,
    Stream stream=*
)

cpdef Column mixed_left_semi_join(
    Table left_keys,
    Table right_keys,
    Table left_conditional,
    Table right_conditional,
    Expression binary_predicate,
    null_equality nulls_equal,
    Stream stream=*
)

cpdef Column mixed_left_anti_join(
    Table left_keys,
    Table right_keys,
    Table left_conditional,
    Table right_conditional,
    Expression binary_predicate,
    null_equality nulls_equal,
    Stream stream=*
)
