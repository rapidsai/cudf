# Copyright (c) 2024, NVIDIA CORPORATION.

from libcpp cimport bool

from cudf._lib.pylibcudf.libcudf.types cimport size_type

from .column cimport Column
from .table cimport Table


ctypedef fused ColumnOrSizeType:
    Column
    size_type

cpdef Table explode_outer(Table, size_type explode_column_idx)

cpdef Column concatenate_rows(Table)

cpdef Column concatenate_list_elements(Column, bool dropna)

cpdef Column extract_list_element(Column, ColumnOrSizeType)
