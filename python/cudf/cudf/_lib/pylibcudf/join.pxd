# Copyright (c) 2024, NVIDIA CORPORATION.

from cudf._lib.pylibcudf.libcudf.types cimport null_equality

from .column cimport Column
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
