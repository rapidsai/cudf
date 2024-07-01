# Copyright (c) 2024, NVIDIA CORPORATION.

from cudf._lib.pylibcudf.libcudf.reduce cimport scan_type

from .aggregation cimport Aggregation
from .column cimport Column
from .scalar cimport Scalar
from .types cimport DataType


cpdef Scalar reduce(Column col, Aggregation agg, DataType data_type)

cpdef Column scan(Column col, Aggregation agg, scan_type inclusive)

cpdef tuple minmax(Column col)
