# Copyright (c) 2023, NVIDIA CORPORATION.

from pyarrow.lib cimport Scalar as pa_Scalar, Table as pa_Table

from .scalar cimport Scalar
from .table cimport Table


cpdef Table from_arrow(
    pa_Table pyarrow_table,
)

cpdef Scalar from_arrow_scalar(
    pa_Scalar pyarrow_scalar,
)
