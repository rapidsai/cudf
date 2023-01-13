# Copyright (c) 2023, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr

from cudf._lib.cpp.table.table cimport table


# TODO: This class represents perhaps the largest argument for divergence from
# pure libcudf-like behavior. It's not clear how to translate C++ ownership
# semantics into something sensible in Python. Having Columns always own data,
# and having pylibcudf Tables just own Python Columns instead of C++ tables
# might be much simpler and more consistent.  Otherwise there are lots of cases
# where we have to think about the possibility of invalidating an object when
# it is used in certain ways, e.g. `Table([col1, col2, ...])` invalidates col1
# and col2.  That feels very unpythonic.
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
