# Copyright (c) 2024-2025, NVIDIA CORPORATION.

from pylibcudf.libcudf.reduce cimport scan_type
from rmm.pylibrmm.stream cimport Stream

from .aggregation cimport Aggregation
from .column cimport Column
from .scalar cimport Scalar
from .types cimport DataType


cpdef Scalar reduce(Column col, Aggregation agg, DataType data_type, Stream stream = *)

cpdef Column scan(Column col, Aggregation agg, scan_type inclusive, Stream stream = *)

cpdef tuple minmax(Column col, Stream stream = *)
