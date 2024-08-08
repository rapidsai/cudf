# Copyright (c) 2024, NVIDIA CORPORATION.

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
