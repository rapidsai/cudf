# Copyright (c) 2023, NVIDIA CORPORATION.

from cython.operator cimport dereference
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from libcpp.vector cimport vector

from rmm._lib.device_buffer cimport DeviceBuffer

from cudf._lib.cpp.column.column cimport column, column_contents
from cudf._lib.cpp.types cimport bitmask_type, offset_type, size_type

from .gpumemoryview cimport gpumemoryview, gpumemoryview_from_device_buffer
from .types cimport DataType
from .utils cimport int_to_bitmask_ptr, int_to_void_ptr


cdef class Column:
    """A container of nullable device data as a column of elements."""
    def __init__(
        self, DataType data_type not None, size_type size, gpumemoryview data,
        gpumemoryview mask, size_type null_count, offset_type offset,
        # TODO: Not sure what the best input is for children, for now just
        # using a List[Column]
        object children
    ):
        self.data_type = data_type
        self.size = size
        self.data = data
        self.mask = mask
        self.null_count = null_count
        self.offset = offset
        self.children = children

    cdef column_view* get_underlying(self):
        cdef const void * data = NULL
        cdef const bitmask_type * null_mask = NULL
        cdef vector[column_view] c_children
        cdef Column child

        if not self._underlying:
            if self.data is not None:
                data = int_to_void_ptr(self.data.ptr)
            if self.mask is not None:
                null_mask = int_to_bitmask_ptr(self.mask.ptr)

            if self.children is not None:
                for child in self.children:
                    c_children.push_back(dereference(child.get_underlying()))

            self._underlying.reset(
                new column_view(
                    self.data_type.c_obj, self.size, data, null_mask,
                    self.null_count, self.offset, c_children
                )
            )
        return self._underlying.get()

    @staticmethod
    cdef Column from_libcudf(unique_ptr[column] libcudf_col):
        cdef DataType dtype = DataType.from_data_type(libcudf_col.get().type())
        cdef size_type size = libcudf_col.get().size()
        cdef size_type null_count = libcudf_col.get().null_count()

        cdef column_contents contents = move(libcudf_col.get().release())

        cdef gpumemoryview data = gpumemoryview_from_device_buffer(
            DeviceBuffer.c_from_unique_ptr(move(contents.data))
        )

        cdef gpumemoryview mask = None
        if null_count > 0:
            mask = gpumemoryview_from_device_buffer(
                DeviceBuffer.c_from_unique_ptr(move(contents.null_mask))
            )

        children = []
        if contents.children.size() != 0:
            for i in range(contents.children.size()):
                children.append(
                    Column.from_libcudf(move(contents.children[i]))
                )

        return Column(
            dtype,
            size,
            data,
            mask,
            null_count,
            # Initial offset when capturing a C++ column is always 0.
            0,
            children,
        )
