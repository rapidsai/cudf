# Copyright (c) 2023, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

from rmm._lib.device_buffer cimport DeviceBuffer

from cudf._lib.cpp.column.column cimport column, column_contents
from cudf._lib.cpp.types cimport offset_type, size_type

from . cimport libcudf_classes
from .gpumemoryview cimport gpumemoryview, gpumemoryview_from_device_buffer
from .types cimport DataType


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

        self._underlying = None

    cpdef libcudf_classes.ColumnView get_underlying(self):
        if self._underlying is None:
            self._underlying = libcudf_classes.ColumnView(
                self.data_type,
                self.size,
                self.data,
                self.mask,
                self.null_count,
                self.offset,
                self.children,
            )
        return self._underlying

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
