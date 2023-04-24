# Copyright (c) 2023, NVIDIA CORPORATION.

from cython.operator cimport dereference
from libcpp cimport bool as cbool
from libcpp.memory cimport make_unique, unique_ptr
from libcpp.utility cimport move

from rmm._lib.device_buffer cimport DeviceBuffer

from cudf._lib.cpp.column.column cimport column, column_contents
from cudf._lib.cpp.types cimport size_type

from .column_view cimport ColumnView


cdef class Column:
    """Wrapper around column."""
    # Initialize released in __cinit__, not __init__, so that it is initialized
    # unconditionally for every object.
    def __cinit__(self):
        self.released = False

    cdef column * get(self) noexcept:
        """Get the underlying column object."""
        return self.c_obj.get()

    cpdef size_type size(self) except -1:
        self._raise_if_released()
        return self.get().size()

    cpdef size_type null_count(self) except -1:
        self._raise_if_released()
        return self.get().null_count()

    cpdef cbool has_nulls(self) except *:
        self._raise_if_released()
        return self.get().has_nulls()

    cpdef ColumnView view(self):
        return ColumnView.from_column_view(self.get().view())

    cdef int _raise_if_released(self) except 1:
        if self.released:
            raise ValueError(
                "Attempted to perform operations on a Column after its "
                "contents have been released."
            )
        return 0

    cpdef ColumnContents release(self):
        """Release the data in this column."""
        cdef column_contents contents = move(self.get().release())
        cdef ColumnContents ret = ColumnContents()
        ret.data = DeviceBuffer.c_from_unique_ptr(move(contents.data))
        ret.null_mask = DeviceBuffer.c_from_unique_ptr(
            move(contents.null_mask)
        )

        cdef Column child
        cdef int i
        ret.children = []

        for i in range(contents.children.size()):
            child = Column()
            child.c_obj.swap(contents.children[i])
            ret.children.append(child)

        self.released = True
        return ret

    @staticmethod
    cdef Column from_column(unique_ptr[column] col):
        cdef Column ret = Column.__new__(Column)
        ret.c_obj.swap(col)
        return ret


# Unfortunately we can't cpdef a staticmethod. Defining an external factory
# separately is the best workaround for now.
# https://github.com/cython/cython/issues/3327
cpdef Column Column_from_ColumnView(ColumnView cv):
    cdef Column ret = Column.__new__(Column)
    cdef unique_ptr[column] c_result
    with nogil:
        c_result = move(make_unique[column](dereference(cv.get())))
    ret.c_obj.swap(c_result)
    return ret
