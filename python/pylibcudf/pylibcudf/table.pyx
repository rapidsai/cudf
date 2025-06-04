# Copyright (c) 2023-2025, NVIDIA CORPORATION.

from cython.operator cimport dereference

from cpython.pycapsule cimport (
    PyCapsule_GetPointer,
    PyCapsule_New,
)

from libcpp.memory cimport unique_ptr, make_unique
from libcpp.utility cimport move
from libcpp.vector cimport vector

from rmm.pylibrmm.stream cimport Stream
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.interop cimport (
    ArrowArray,
    ArrowArrayStream,
    ArrowDeviceArray,
    ArrowSchema,
    arrow_table,
    column_metadata,
    to_arrow_device_raw,
    to_arrow_host_raw,
    to_arrow_schema_raw,
)
from pylibcudf.libcudf.table.table cimport table

from .column cimport Column
from .types cimport DataType
from .utils cimport _get_stream
from pylibcudf._interop_helpers cimport (
    _release_schema,
    _release_array,
    _release_device_array,
    _metadata_to_libcudf,
)
from ._interop_helpers import ArrowLike, ColumnMetadata

__all__ = ["Table"]


cdef class _ArrowTableHolder:
    """A holder for an Arrow table for gpumemoryview lifetime management."""
    cdef unique_ptr[arrow_table] tbl


cdef class Table:
    """A list of columns of the same size.

    Parameters
    ----------
    columns : list
        The columns in this table.
    """
    __hash__ = None

    def __init__(self, list columns):
        if not all(isinstance(c, Column) for c in columns):
            raise ValueError("All columns must be pylibcudf Column objects")
        self._columns = columns

    @staticmethod
    def from_arrow(obj: ArrowLike, dtype: DataType | None = None) -> Table:
        """
        Create a Table from an Arrow-like object using the Arrow C data interface.

        This method supports constructing a `pylibcudf.Table` from an object that
        implements one of the Arrow C data interface protocols, such as:

        - `__arrow_c_device_array__`: Returns a tuple of (ArrowSchema, ArrowDeviceArray)
          representing columnar device memory.
        - `__arrow_c_stream__`: Returns an ArrowArrayStream pointer representing
          a stream of host columnar batches.
        - `__arrow_c_device_stream__`: Not yet implemented.
        - `__arrow_c_array__`: Not yet implemented.

        Parameters
        ----------
        obj : Arrow-like type
            An object implementing one of the Arrow C data interface methods.
        dtype: DataType
            The pylibcudf data type.

        Returns
        -------
        Table
            A Table constructed from the Arrow-like input.

        Raises
        ------
        NotImplementedError
            If the input is a device or host stream not yet supported.
            If the dtype argument is not None.
        ValueError
            If the input does not implement a supported Arrow C interface.
        """
        if dtype is not None:
            raise NotImplementedError(
                "Creating a Table with the dtype argument specified."
            )
        cdef ArrowSchema* c_schema
        cdef ArrowDeviceArray* c_array
        cdef _ArrowTableHolder result
        cdef unique_ptr[arrow_table] c_result
        if hasattr(obj, "__arrow_c_device_array__"):
            schema, array = obj.__arrow_c_device_array__()
            c_schema = <ArrowSchema*>PyCapsule_GetPointer(schema, "arrow_schema")
            c_array = (
                <ArrowDeviceArray*>PyCapsule_GetPointer(array, "arrow_device_array")
            )

            result = _ArrowTableHolder()
            with nogil:
                c_result = make_unique[arrow_table](
                    move(dereference(c_schema)), move(dereference(c_array))
                )
            result.tbl.swap(c_result)

            return Table.from_table_view_of_arbitrary(result.tbl.get().view(), result)
        elif hasattr(obj, "__arrow_c_stream__"):
            stream = obj.__arrow_c_stream__()
            c_stream = (
                <ArrowArrayStream*>PyCapsule_GetPointer(stream, "arrow_array_stream")
            )

            result = _ArrowTableHolder()
            with nogil:
                c_result = make_unique[arrow_table](move(dereference(c_stream)))
            result.tbl.swap(c_result)

            return Table.from_table_view_of_arbitrary(result.tbl.get().view(), result)
        elif hasattr(obj, "__arrow_c_device_stream__"):
            # TODO: When we add support for this case, it should be moved above
            # the __arrow_c_stream__ case since we should prioritize device
            # data if possible.
            raise NotImplementedError("Device streams not yet supported")
        elif hasattr(obj, "__arrow_c_array__"):
            raise NotImplementedError("Arrow host arrays not yet supported")
        else:
            raise ValueError("Invalid Arrow-like object")

    cdef table_view view(self) nogil:
        """Generate a libcudf table_view to pass to libcudf algorithms.

        This method is for pylibcudf's functions to use to generate inputs when
        calling libcudf algorithms, and should generally not be needed by users
        (even direct pylibcudf Cython users).
        """
        # TODO: Make c_columns a class attribute that is updated along with
        # self._columns whenever new columns are added or columns are removed.
        cdef vector[column_view] c_columns

        with gil:
            for col in self._columns:
                c_columns.push_back((<Column> col).view())

        return table_view(c_columns)

    @staticmethod
    cdef Table from_libcudf(unique_ptr[table] libcudf_tbl, Stream stream=None):
        """Create a Table from a libcudf table.

        This method is for pylibcudf's functions to use to ingest outputs of
        calling libcudf algorithms, and should generally not be needed by users
        (even direct pylibcudf Cython users).
        """
        cdef vector[unique_ptr[column]] c_columns = dereference(libcudf_tbl).release()

        cdef vector[unique_ptr[column]].size_type i
        stream = _get_stream(stream)
        return Table([
            Column.from_libcudf(move(c_columns[i]), stream)
            for i in range(c_columns.size())
        ])

    @staticmethod
    cdef Table from_table_view(const table_view& tv, Table owner):
        """Create a Table from a libcudf table_view into a Table owner.

        This method accepts shared ownership of the underlying data from the owner.

        This method is for pylibcudf's functions to use to ingest outputs of
        calling libcudf algorithms, and should generally not be needed by users
        (even direct pylibcudf Cython users).
        """
        cdef int i
        return Table([
            Column.from_column_view(tv.column(i), owner.columns()[i])
            for i in range(tv.num_columns())
        ])

    # Ideally this function would simply be handled via a fused type in
    # from_table_view, but this does not work due to
    # https://github.com/cython/cython/issues/6740
    @staticmethod
    cdef Table from_table_view_of_arbitrary(const table_view& tv, object owner):
        """Create a Table from a libcudf table_view into an arbitrary owner.

        This method accepts shared ownership of the underlying data from the owner.
        Since the owner may be any arbitrary object, every buffer view that is part of
        the table (each column and all of their children) share ownership of the same
        buffer since they do not have the information available to choose to only own
        subsets of it.

        This method is for pylibcudf's functions to use to ingest outputs of
        calling libcudf algorithms, and should generally not be needed by users
        (even direct pylibcudf Cython users).
        """
        # For efficiency, prohibit calling this overload with a Table owner.
        assert not isinstance(owner, Table)
        cdef int i
        return Table([
            Column.from_column_view_of_arbitrary(tv.column(i), owner)
            for i in range(tv.num_columns())
        ])

    cpdef int num_columns(self):
        """The number of columns in this table."""
        return len(self._columns)

    cpdef int num_rows(self):
        """The number of rows in this table."""
        if self.num_columns() == 0:
            return 0
        return self._columns[0].size()

    cpdef list columns(self):
        """The columns in this table."""
        return self._columns

    cpdef tuple shape(self):
        """The shape of this table"""
        return (self.num_rows(), self.num_columns())

    def _to_schema(self, metadata=None):
        """Create an Arrow schema from this table."""
        if metadata is None:
            metadata = [
                col._create_nested_column_metadata() for col in self.columns()
            ]
        else:
            metadata = [
                ColumnMetadata(m) if isinstance(m, str) else m for m in metadata
            ]

        cdef vector[column_metadata] c_metadata
        c_metadata.reserve(len(metadata))
        for meta in metadata:
            c_metadata.push_back(_metadata_to_libcudf(meta))

        cdef ArrowSchema* raw_schema_ptr
        with nogil:
            raw_schema_ptr = to_arrow_schema_raw(self.view(), c_metadata)

        return PyCapsule_New(<void*>raw_schema_ptr, "arrow_schema", _release_schema)

    def _to_host_array(self):
        cdef ArrowArray* raw_host_array_ptr
        with nogil:
            raw_host_array_ptr = to_arrow_host_raw(self.view())

        return PyCapsule_New(<void*>raw_host_array_ptr, "arrow_array", _release_array)

    def _to_device_array(self):
        cdef ArrowDeviceArray* raw_device_array_ptr
        with nogil:
            raw_device_array_ptr = to_arrow_device_raw(self.view(), self)

        return PyCapsule_New(
            <void*>raw_device_array_ptr,
            "arrow_device_array",
            _release_device_array
        )

    def __arrow_c_array__(self, requested_schema=None):
        if requested_schema is not None:
            raise ValueError("pylibcudf.Table does not support alternative schema")

        return self._to_schema(), self._to_host_array()

    def __arrow_c_device_array__(self, requested_schema=None, **kwargs):
        if requested_schema is not None:
            raise ValueError("pylibcudf.Table does not support alternative schema")

        non_default_kwargs = [
            name for name, value in kwargs.items() if value is not None
        ]
        if non_default_kwargs:
            raise NotImplementedError(
                f"Received unsupported keyword argument(s): {non_default_kwargs}"
            )

        return self._to_schema(), self._to_device_array()
