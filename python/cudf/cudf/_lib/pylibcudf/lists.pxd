# Copyright (c) 2024, NVIDIA CORPORATION.

from libcpp cimport bool

from cudf._lib.pylibcudf.libcudf.types cimport size_type

from .column cimport Column
from .scalar cimport Scalar
from .table cimport Table

ctypedef fused ColumnOrScalar:
    Column
    Scalar

cpdef Table explode_outer(Table, size_type explode_column_idx)

cpdef Column concatenate_rows(Table)

cpdef Column concatenate_list_elements(Column, bool dropna)

cpdef Column contains(Column, ColumnOrScalar)

# cpdef Column contains_nulls(Column)

# ctypedef Column index_of(Column, ColumnOrScalar)

# from cudf._lib.pylibcudf.libcudf.binaryop import \
#     binary_operator as BinaryOperator  # no-cython-lint
# from cudf._lib.pylibcudf.libcudf.lists.contains cimport duplicate_find_option
