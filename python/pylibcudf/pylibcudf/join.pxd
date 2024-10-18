# Copyright (c) 2024, NVIDIA CORPORATION.

from pylibcudf.libcudf.types cimport null_equality

from .column cimport Column
from .expressions cimport Expression
from .table cimport Table


cpdef tuple inner_join(
    Table left_keys,
    Table right_keys,
    null_equality nulls_equal
)

cpdef tuple left_join(
    Table left_keys,
    Table right_keys,
    null_equality nulls_equal
)

cpdef tuple full_join(
    Table left_keys,
    Table right_keys,
    null_equality nulls_equal
)

cpdef Column left_semi_join(
    Table left_keys,
    Table right_keys,
    null_equality nulls_equal
)

cpdef Column left_anti_join(
    Table left_keys,
    Table right_keys,
    null_equality nulls_equal
)

cpdef Table cross_join(Table left, Table right)

cpdef tuple conditional_inner_join(
    Table left,
    Table right,
    Expression binary_predicate,
)

cpdef tuple conditional_left_join(
    Table left,
    Table right,
    Expression binary_predicate,
)

cpdef tuple conditional_full_join(
    Table left,
    Table right,
    Expression binary_predicate,
)

cpdef Column conditional_left_semi_join(
    Table left,
    Table right,
    Expression binary_predicate,
)

cpdef Column conditional_left_anti_join(
    Table left,
    Table right,
    Expression binary_predicate,
)

cpdef tuple mixed_inner_join(
    Table left_keys,
    Table right_keys,
    Table left_conditional,
    Table right_conditional,
    Expression binary_predicate,
    null_equality nulls_equal
)

cpdef tuple mixed_left_join(
    Table left_keys,
    Table right_keys,
    Table left_conditional,
    Table right_conditional,
    Expression binary_predicate,
    null_equality nulls_equal
)

cpdef tuple mixed_full_join(
    Table left_keys,
    Table right_keys,
    Table left_conditional,
    Table right_conditional,
    Expression binary_predicate,
    null_equality nulls_equal
)

cpdef Column mixed_left_semi_join(
    Table left_keys,
    Table right_keys,
    Table left_conditional,
    Table right_conditional,
    Expression binary_predicate,
    null_equality nulls_equal
)

cpdef Column mixed_left_anti_join(
    Table left_keys,
    Table right_keys,
    Table left_conditional,
    Table right_conditional,
    Expression binary_predicate,
    null_equality nulls_equal
)
