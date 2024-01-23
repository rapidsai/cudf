# Copyright (c) 2023-2024, NVIDIA CORPORATION.

from cudf._lib.cpp.interop cimport column_metadata


cdef class ColumnMetadata:
    """Metadata associated with a column.

    This is the Cython representation of :cpp:class:`cudf::column_metadata`.

    Parameters
    ----------
    id : TypeId
        The type's identifier
    scale : int
        The scale associated with the data. Only used for decimal data types.
    """
    def __init__(self, name):
        self.name = name
        self.children_meta = []

    cdef column_metadata to_libcudf(self):
        """Convert to C++ column_metadata.

        Since this class is mutable and cheap, it is easier to create the C++
        object on the fly rather than have it directly backing the storage for
        the Cython class.
        """
        cdef column_metadata c_metadata
        cdef ColumnMetadata child_meta
        c_metadata.name = self.name.encode()
        for child_meta in self.children_meta:
            c_metadata.children_meta.push_back(child_meta.to_libcudf())
        return c_metadata
