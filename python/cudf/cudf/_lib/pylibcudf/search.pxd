# Copyright (c) 2024, NVIDIA CORPORATION.

# from libcpp cimport bool
#
# from cudf._lib.cpp.aggregation cimport rank_method
# from cudf._lib.cpp.types cimport null_order, null_policy, order, size_type

from .column cimport Column
from .table cimport Table


cpdef Column lower_bound(
    Table haystack,
    Table needles,
    list column_order,
    list null_precedence,
)

cpdef Column upper_bound(
    Table haystack,
    Table needles,
    list column_order,
    list null_precedence,
)

cpdef Column contains(Column haystack, Column needles)
