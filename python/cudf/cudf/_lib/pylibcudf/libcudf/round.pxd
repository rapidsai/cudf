# Copyright (c) 2021-2024, NVIDIA CORPORATION.

from libc.stdint cimport int32_t
from libcpp.memory cimport unique_ptr

from cudf._lib.pylibcudf.libcudf.column.column cimport column
from cudf._lib.pylibcudf.libcudf.column.column_view cimport column_view


cdef extern from "cudf/round.hpp" namespace "cudf" nogil:

    ctypedef enum rounding_method "cudf::rounding_method":
        HALF_UP "cudf::rounding_method::HALF_UP"
        HALF_EVEN "cudf::rounding_method::HALF_EVEN"

    cdef unique_ptr[column] round (
        const column_view& input,
        int32_t decimal_places,
        rounding_method method,
    ) except +
