# Copyright (c) 2025, NVIDIA CORPORATION.

from cpython.pycapsule cimport PyCapsule_GetPointer

from pylibcudf.libcudf.interop cimport (
    ArrowArray,
    ArrowSchema,
    column_metadata,
    release_arrow_array_raw,
    release_arrow_schema_raw,
)

from dataclasses import dataclass, field


@dataclass
class ColumnMetadata:
    """Metadata associated with a column.

    This is the Python representation of :cpp:class:`cudf::column_metadata`.
    """
    name: str = ""
    children_meta: list[ColumnMetadata] = field(default_factory=list)


cdef void _release_schema(object schema_capsule) noexcept:
    """Release the ArrowSchema object stored in a PyCapsule."""
    cdef ArrowSchema* schema = <ArrowSchema*>PyCapsule_GetPointer(
        schema_capsule, 'arrow_schema'
    )
    release_arrow_schema_raw(schema)


cdef void _release_array(object array_capsule) noexcept:
    """Release the ArrowArray object stored in a PyCapsule."""
    cdef ArrowArray* array = <ArrowArray*>PyCapsule_GetPointer(
        array_capsule, 'arrow_array'
    )
    release_arrow_array_raw(array)


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
