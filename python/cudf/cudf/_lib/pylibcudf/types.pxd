# Copyright (c) 2023-2024, NVIDIA CORPORATION.

from libc.stdint cimport int32_t
from libcpp cimport bool as cbool

from cudf._lib.pylibcudf.libcudf.types cimport (
    data_type,
    interpolation,
    mask_state,
    nan_equality,
    nan_policy,
    null_equality,
    null_order,
    null_policy,
    order,
    sorted,
    type_id,
)

from cudf._lib.pylibcudf.libcudf.types import \
    interpolation as Interpolation  # no-cython-lint
from cudf._lib.pylibcudf.libcudf.types import \
    null_order as NullOrder  # no-cython-lint
from cudf._lib.pylibcudf.libcudf.types import \
    sorted as Sorted  # no-cython-lint
from cudf._lib.pylibcudf.libcudf.types import order as Order  # no-cython-lint


cdef class DataType:
    cdef data_type c_obj

    cpdef type_id id(self)
    cpdef int32_t scale(self)

    @staticmethod
    cdef DataType from_libcudf(data_type dt)
