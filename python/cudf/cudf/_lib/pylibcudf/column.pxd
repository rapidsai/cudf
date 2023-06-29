# Copyright (c) 2023, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr

from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.types cimport offset_type, size_type

from .gpumemoryview cimport gpumemoryview
from .types cimport DataType


cdef class Column:
    cdef:
        # Core data
        DataType data_type
        size_type size
        gpumemoryview data
        gpumemoryview mask
        size_type null_count
        offset_type offset
        # children: List[Column]
        list children

        # Internals
        unique_ptr[column_view] _view

    # We store a unique pointer and return a raw pointer to that object here
    # rather than storing a column_view and returning by value each time to
    # ensure that we do not make an unnecessarily large number of copies.
    # Cython-generated C++ code will always include numerous intermediate
    # assignments that will prevent copy elision like you would see in the
    # equivalent C++ code. Moreover, this function must access a Python list
    # (of children) to construct the column_view, which means it cannot be
    # entirely nogil. The output of this function will be passed into libcudf
    # function calls that invariably happen inside nogil blocks, which
    # guarantees that the generated code will not be amenable to copy elision.
    # As such, this pointer-based approach is the only way to avoid a large
    # number of additional copies from being made.
    cdef column_view* view(self)

    @staticmethod
    cdef Column from_libcudf(unique_ptr[column] libcudf_col)
