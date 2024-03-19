# Copyright (c) 2023-2024, NVIDIA CORPORATION.

from cython.operator cimport dereference
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

from rmm._lib.device_buffer cimport DeviceBuffer

from cudf._lib.cpp.column.column cimport column, column_contents
from cudf._lib.cpp.column.column_factories cimport make_column_from_scalar
from cudf._lib.cpp.scalar.scalar cimport scalar
from cudf._lib.cpp.types cimport size_type

from .gpumemoryview cimport gpumemoryview
from .scalar cimport Scalar
from .types cimport DataType, type_id
from .utils cimport int_to_bitmask_ptr, int_to_void_ptr


cdef class Column:
    """A container of nullable device data as a column of elements.

    This class is an implementation of `Arrow columnar data specification
    <https://arrow.apache.org/docs/format/Columnar.html>`__ for data stored on
    GPUs. It relies on Python memoryview-like semantics to maintain shared
    ownership of the data it is constructed with, so any input data may also be
    co-owned by other data structures. The Column is designed to be operated on
    using algorithms backed by libcudf.

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
        if not all(isinstance(c, Column) for c in children):
            raise ValueError("All children must be pylibcudf Column objects")
        self._data_type = data_type
        self._size = size
        self._data = data
        self._mask = mask
        self._null_count = null_count
        self._offset = offset
        self._children = children
        self._num_children = len(children)

    cdef column_view view(self) nogil:
        """Generate a libcudf column_view to pass to libcudf algorithms.

        This method is for pylibcudf's functions to use to generate inputs when
        calling libcudf algorithms, and should generally not be needed by users
        (even direct pylibcudf Cython users).
        """
        cdef const void * data = NULL
        cdef const bitmask_type * null_mask = NULL

        if self._data is not None:
            data = int_to_void_ptr(self._data.ptr)
        if self._mask is not None:
            null_mask = int_to_bitmask_ptr(self._mask.ptr)

        # TODO: Check if children can ever change. If not, this could be
        # computed once in the constructor and always be reused.
        cdef vector[column_view] c_children
        with gil:
            if self._children is not None:
                for child in self._children:
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
            self._data_type.c_obj, self._size, data, null_mask,
            self._null_count, self._offset, c_children
        )

    cdef mutable_column_view mutable_view(self) nogil:
        """Generate a libcudf mutable_column_view to pass to libcudf algorithms.

        This method is for pylibcudf's functions to use to generate inputs when
        calling libcudf algorithms, and should generally not be needed by users
        (even direct pylibcudf Cython users).
        """
        cdef void * data = NULL
        cdef bitmask_type * null_mask = NULL

        if self._data is not None:
            data = int_to_void_ptr(self._data.ptr)
        if self._mask is not None:
            null_mask = int_to_bitmask_ptr(self._mask.ptr)

        cdef vector[mutable_column_view] c_children
        with gil:
            if self._children is not None:
                for child in self._children:
                    # See the view method for why this needs to be cast.
                    c_children.push_back((<Column> child).mutable_view())

        return mutable_column_view(
            self._data_type.c_obj, self._size, data, null_mask,
            self._null_count, self._offset, c_children
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

    @staticmethod
    cdef Column from_column_view(const column_view& cv, Column owner):
        """Create a Column from a libcudf column_view.

        This method accepts shared ownership of the underlying data from the
        owner and relies on the offset from the view.

        This method is for pylibcudf's functions to use to ingest outputs of
        calling libcudf algorithms, and should generally not be needed by users
        (even direct pylibcudf Cython users).
        """
        cdef DataType dtype = DataType.from_libcudf(cv.type())
        cdef size_type size = cv.size()
        cdef size_type null_count = cv.null_count()

        children = []
        if cv.num_children() != 0:
            for i in range(cv.num_children()):
                children.append(
                    Column.from_column_view(cv.child(i), owner.child(i))
                )

        return Column(
            dtype,
            size,
            owner._data,
            owner._mask,
            null_count,
            cv.offset(),
            children,
        )

    @staticmethod
    def from_scalar(Scalar slr, size_type size):
        """Create a Column from a Scalar.

        Parameters
        ----------
        slr : Scalar
            The scalar to create a column from.
        size : size_type
            The number of elements in the column.

        Returns
        -------
        Column
            A Column containing the scalar repeated `size` times.
        """
        cdef const scalar* c_scalar = slr.get()
        cdef unique_ptr[column] c_result
        with nogil:
            c_result = move(make_column_from_scalar(dereference(c_scalar), size))
        return Column.from_libcudf(move(c_result))

    cpdef DataType type(self):
        """The type of data in the column."""
        return self._data_type

    cpdef Column child(self, size_type index):
        """Get a child column of this column.

        Parameters
        ----------
        index : size_type
            The index of the child column to get.

        Returns
        -------
        Column
            The child column.
        """
        return self._children[index]

    cpdef size_type num_children(self):
        """The number of children of this column."""
        return self._num_children

    cpdef ListColumnView list_view(self):
        """Accessor for methods of a Column that are specific to lists."""
        return ListColumnView(self)

    cpdef gpumemoryview data(self):
        """The data buffer of the column."""
        return self._data

    cpdef gpumemoryview null_mask(self):
        """The null mask of the column."""
        return self._mask

    cpdef size_type size(self):
        """The number of elements in the column."""
        return self._size

    cpdef size_type offset(self):
        """The offset of the column."""
        return self._offset

    cpdef size_type null_count(self):
        """The number of null elements in the column."""
        return self._null_count

    cpdef list children(self):
        """The children of the column."""
        return self._children


cdef class ListColumnView:
    """Accessor for methods of a Column that are specific to lists."""
    def __init__(self, Column col):
        if col.type().id() != type_id.LIST:
            raise TypeError("Column is not a list type")
        self._column = col

    cpdef child(self):
        """The data column of the underlying list column."""
        return self._column.child(1)

    cpdef offsets(self):
        """The offsets column of the underlying list column."""
        return self._column.child(1)
