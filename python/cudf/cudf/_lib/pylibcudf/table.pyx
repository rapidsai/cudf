# Copyright (c) 2023, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr

from cudf._lib.cpp.table.table cimport table


cdef class Table:
    """Wrapper around table."""
    cdef table * get(self) noexcept:
        """Get the underlying table object."""
        return self.c_obj.get()

    @staticmethod
    cdef from_table(unique_ptr[table] tbl):
        cdef Table ret = Table.__new__(Table)
        ret.c_obj.swap(tbl)
        return ret
