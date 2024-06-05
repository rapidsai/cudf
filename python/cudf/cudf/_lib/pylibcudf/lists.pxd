# Copyright (c) 2024, NVIDIA CORPORATION.

from cudf._lib.pylibcudf.libcudf.types cimport size_type

from .column cimport Column
from .table cimport Table


cpdef Table explode_outer(Table, size_type explode_column_idx)

cpdef Column concatenate_rows(Table)
