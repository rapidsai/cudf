# Copyright (c) 2023-2025, NVIDIA CORPORATION.

from cython.operator cimport dereference

from cpython.pycapsule cimport (
    PyCapsule_GetPointer,
    PyCapsule_New,
)

from libc.stdlib cimport free
from libcpp.memory cimport unique_ptr, make_unique
from libcpp.utility cimport move
from libcpp.vector cimport vector

from functools import singledispatchmethod

from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.table.table cimport table

from pylibcudf.libcudf.interop cimport (
    ArrowArray,
    ArrowArrayStream,
    ArrowSchema,
    arrow_table,
    column_metadata,
    to_arrow_host_raw,
    to_arrow_schema_raw,
)

from .column cimport Column

__all__ = ["Table"]


# TODO: Add a strong type here on the ColumnMetadata input
cdef column_metadata _metadata_to_libcudf(metadata):
    """Convert a ColumnMetadata object to C++ column_metadata.

    Since this class is mutable and cheap, it is easier to create the C++
    object on the fly rather than have it directly backing the storage for
    the Cython class. Additionally, this structure restricts the dependency
    on C++ types to just within this module, allowing us to make the module a
    pure Python module (from an import sense, i.e. no pxd declarations).
    """
    cdef column_metadata c_metadata
    c_metadata.name = metadata.name.encode()
    for child_meta in metadata.children_meta:
        c_metadata.children_meta.push_back(_metadata_to_libcudf(child_meta))
    return c_metadata


cdef void _release_schema(object schema_capsule) noexcept:
    """Release the ArrowSchema object stored in a PyCapsule."""
    cdef ArrowSchema* schema = <ArrowSchema*>PyCapsule_GetPointer(
        schema_capsule, 'arrow_schema'
    )
    if schema.release != NULL:
        schema.release(schema)

    free(schema)


cdef void _release_array(object array_capsule) noexcept:
    """Release the ArrowArray object stored in a PyCapsule."""
    cdef ArrowArray* array = <ArrowArray*>PyCapsule_GetPointer(
        array_capsule, 'arrow_array'
    )
    if array.release != NULL:
        array.release(array)

    free(array)


class _ArrowLikeMeta(type):
    # Unfortunately we cannot separate stream and array via singledispatch because the
    # dispatch will often be ambiguous when objects expose both protocols.
    def __subclasscheck__(cls, other):
        return (
            hasattr(other, "__arrow_c_stream__")
            or hasattr(other, "__arrow_c_array__")
        )


class _ArrowLike(metaclass=_ArrowLikeMeta):
    pass


cdef class _ArrowTableHolder:
    cdef unique_ptr[arrow_table] tbl


cdef class Table:
    """A list of columns of the same size.

    Parameters
    ----------
    columns : list
        The columns in this table.
    """
    def __init__(self, obj):
        self._init(obj)

    __hash__ = None

    @singledispatchmethod
    def _init(self, obj):
        raise ValueError(f"Invalid input type {type(obj)}")

    @_init.register(list)
    def _(self, list columns):
        if not all(isinstance(c, Column) for c in columns):
            raise ValueError("All columns must be pylibcudf Column objects")
        self._columns = columns

    @_init.register(_ArrowLike)
    def _(self, arrow_like):
        cdef ArrowArrayStream* c_stream
        cdef _ArrowTableHolder result
        cdef unique_ptr[arrow_table] c_result
        if hasattr(arrow_like, "__arrow_c_stream__"):
            stream = arrow_like.__arrow_c_stream__()
            c_stream = (
                <ArrowArrayStream*>PyCapsule_GetPointer(stream, "arrow_array_stream")
            )

            result = _ArrowTableHolder()
            with nogil:
                c_result = make_unique[arrow_table](move(dereference(c_stream)))
            result.tbl.swap(c_result)

            tmp = Table.from_table_view_of_arbitrary(result.tbl.get().view(), result)
            self._columns = tmp.columns()
        elif hasattr(arrow_like, "__arrow_c_array__"):
            raise NotImplementedError("arrays not yet supported")

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
    cdef Table from_libcudf(unique_ptr[table] libcudf_tbl):
        """Create a Table from a libcudf table.

        This method is for pylibcudf's functions to use to ingest outputs of
        calling libcudf algorithms, and should generally not be needed by users
        (even direct pylibcudf Cython users).
        """
        cdef vector[unique_ptr[column]] c_columns = dereference(libcudf_tbl).release()

        cdef vector[unique_ptr[column]].size_type i
        return Table([
            Column.from_libcudf(move(c_columns[i]))
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

    @staticmethod
    def _create_nested_column_metadata(Column col):
        # TODO: We'll need to reshuffle where things are defined to avoid circular
        # imports. For now, we'll just import this inline. We should be able to avoid
        # circularity altogether by simply not needing ColumnMetadata at all in the
        # future and just using the schema directly, so we can consider that approach.
        from pylibcudf.interop import ColumnMetadata
        return ColumnMetadata(
            children_meta=[
                Table._create_nested_column_metadata(child) for child in col.children()
            ]
        )

    def _to_schema(self, metadata=None):
        """Create an Arrow schema from this table."""
        # TODO: We'll need to reshuffle where things are defined to avoid circular
        # imports. For now, we'll just import this inline. We should be able to avoid
        # circularity altogether by simply not needing ColumnMetadata at all in the
        # future and just using the schema directly, so we can consider that approach.
        from pylibcudf.interop import ColumnMetadata
        if metadata is None:
            metadata = [
                Table._create_nested_column_metadata(col) for col in self.columns()
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

        return PyCapsule_New(<void*>raw_schema_ptr, 'arrow_schema', _release_schema)

    def _to_host_array(self):
        cdef ArrowArray* raw_host_array_ptr
        with nogil:
            raw_host_array_ptr = to_arrow_host_raw(self.view())

        return PyCapsule_New(<void*>raw_host_array_ptr, "arrow_array", _release_array)

    def __arrow_c_array__(self, requested_schema=None):
        if requested_schema is not None:
            raise ValueError("pylibcudf.Table does not support alternative schema")

        # For the host array protocol the capsules own the data.
        ret = self._to_schema(), self._to_host_array()
        return ret
