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

cpdef Column contains_nulls(Column)

cpdef Column index_of(Column, ColumnOrScalar, bool)

cpdef Column sequences(Column, Column, Column steps = *)
