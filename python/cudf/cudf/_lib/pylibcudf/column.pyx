# Copyright (c) 2023, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

from rmm._lib.device_buffer cimport DeviceBuffer

from cudf._lib.cpp.column.column cimport column, column_contents
from cudf._lib.cpp.types cimport size_type

from .gpumemoryview cimport gpumemoryview
from .types cimport DataType
from .utils cimport int_to_bitmask_ptr, int_to_void_ptr


cdef class Column:
    """A container of nullable device data as a column of elements.

    This class is an implementation of [Arrow columnar data
    specification](https://arrow.apache.org/docs/format/Columnar.html) for data
    stored on GPUs. It relies on Python memoryview-like semantics to maintain
    shared ownership of the data it is constructed with, so any input data may
    also be co-owned by other data structures. The Column is designed to be
    operated on using algorithms backed by libcudf.

    Parameters
    ----------
    data_type : DataType
        The type of data in the column.
    size : size_type
        The number of rows in the column.
    data : gpumemoryview
        The data the column will refer to.
    mask : gpumemoryview
        The null mask for the column.
    null_count : int
        The number of null rows in the column.
    offset : int
        The offset into the data buffer where the column's data begins.
    children : list
        The children of this column if it is a compound column type.
    """
    def __init__(
        self, DataType data_type not None, size_type size, gpumemoryview data,
        gpumemoryview mask, size_type null_count, size_type offset,
        list children
    ):
        self.data_type = data_type
        self.size = size
        self.data = data
        self.mask = mask
        self.null_count = null_count
        self.offset = offset
        self.children = children

    cdef column_view view(self) nogil:
        """Generate a libcudf column_view to pass to libcudf algorithms.

        This method is for pylibcudf's functions to use to generate inputs when
        calling libcudf algorithms, and should generally not be needed by users
        (even direct pylibcudf Cython users).
        """
        cdef const void * data = NULL
        cdef const bitmask_type * null_mask = NULL

        if self.data is not None:
            data = int_to_void_ptr(self.data.ptr)
        if self.mask is not None:
            null_mask = int_to_bitmask_ptr(self.mask.ptr)

        # TODO: Check if children can ever change. If not, this could be
        # computed once in the constructor and always be reused.
        cdef vector[column_view] c_children
        with gil:
            if self.children is not None:
                for child in self.children:
                    # Need to cast to Column here so that Cython knows that
                    # `view` returns a typed object, not a Python object. We
                    # cannot use a typed variable for `child` because cdef
                    # declarations cannot be inside nested blocks (`if` or
                    # `with` blocks) so we cannot declare it inside the `with
                    # gil` block, but we also cannot declare it outside the
                    # `with gil` block because it is erroneous to declare a
                    # variable of a cdef class type in a `nogil` context (which
                    # this whole function is).
                    c_children.push_back((<Column> child).view())

        return column_view(
            self.data_type.c_obj, self.size, data, null_mask,
            self.null_count, self.offset, c_children
        )

    @staticmethod
    cdef Column from_libcudf(unique_ptr[column] libcudf_col):
        """Create a Column from a libcudf column.

        This method is for pylibcudf's functions to use to ingest outputs of
        calling libcudf algorithms, and should generally not be needed by users
        (even direct pylibcudf Cython users).
        """
        cdef DataType dtype = DataType.from_libcudf(libcudf_col.get().type())
        cdef size_type size = libcudf_col.get().size()
        cdef size_type null_count = libcudf_col.get().null_count()

        cdef column_contents contents = move(libcudf_col.get().release())

        # Note that when converting to cudf Column objects we'll need to pull
        # out the base object.
        cdef gpumemoryview data = gpumemoryview(
            DeviceBuffer.c_from_unique_ptr(move(contents.data))
        )

        cdef gpumemoryview mask = None
        if null_count > 0:
            mask = gpumemoryview(
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
