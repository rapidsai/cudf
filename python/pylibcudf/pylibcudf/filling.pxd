# Copyright (c) 2024, NVIDIA CORPORATION.
from pylibcudf.libcudf.types cimport size_type

from .column cimport Column
from .scalar cimport Scalar
from .table cimport Table

ctypedef fused ColumnOrSize:
    Column
    size_type

cpdef Column fill(
    Column destination,
    size_type begin,
    size_type end,
    Scalar value,
)

cpdef void fill_in_place(
    Column destination,
    size_type c_begin,
    size_type c_end,
    Scalar value,
)

cpdef Column sequence(
    size_type size,
    Scalar init,
    Scalar step,
)

cpdef Table repeat(
    Table input_table,
    ColumnOrSize count
)

cpdef Column calendrical_month_sequence(
    size_type n,
    Scalar init,
    size_type months,
)
