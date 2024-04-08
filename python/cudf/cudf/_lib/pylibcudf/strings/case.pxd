# Copyright (c) 2024, NVIDIA CORPORATION.

from cudf._lib.pylibcudf.column cimport Column


cpdef Column to_lower(Column input)
cpdef Column to_upper(Column input)
cpdef Column swapcase(Column input)
