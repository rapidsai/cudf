# Copyright (c) 2023, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

from cudf._lib.cpp.column.column cimport column
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
        # TODO: The move here seems a bit surprising to a Cython user. But it's
        # good C++ practice to show that we're taking ownership using
        # `unique_ptr` in the signature, so maybe it's fine.
        cdef libcudf_classes.Column col = libcudf_classes.Column.from_column(
            move(libcudf_col)
        )

        cdef DataType dtype = col.type()
        cdef size_type size = col.size()
        cdef size_type null_count = col.null_count()

        cdef libcudf_classes.ColumnContents contents = col.release()

        cdef gpumemoryview data = gpumemoryview_from_device_buffer(
            contents.data
        )
        cdef gpumemoryview mask = None
        if null_count > 0:
            mask = gpumemoryview_from_device_buffer(contents.null_mask)

        children = []
        cdef libcudf_classes.Column child
        for child in contents.children:
            # TODO: I don't like that I'm accessing (and moving!) c_obj here.
            # Will probably want to find a cleaner approach, but for now I'm OK
            # with it since it should be easier to think about this once the
            # rest of pylibcudf is switched over to using the new classes
            # anyway.
            children.append(Column.from_libcudf(move(child.c_obj)))

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
