# Copyright (c) 2024, NVIDIA CORPORATION.

from libcpp.string cimport string
from libcpp.utility cimport move
from libcpp.vector cimport vector

from cudf._lib.pylibcudf.libcudf.io.types cimport (
    host_buffer,
    source_info,
    table_with_metadata,
)

import errno
import io
import os


cdef class TableWithMetadata:
    """A container holding a table and its associated metadata
    (e.g. column names)

    For details, see :cpp:class:`cudf::io::table_with_metadata`.
    """

    @property
    def columns(self):
        """
        Return a list containing the columns of the table
        """
        return self.tbl.columns()

    @property
    def column_names(self):
        """
        Return a list containing the column names of the table
        """
        cdef list names = []
        for col_info in self.metadata.schema_info:
            # TODO: Handle nesting (columns with child columns)
            assert col_info.children.size() == 0, "Child column names are not handled!"
            names.append(col_info.name.decode())
        return names

    @staticmethod
    cdef TableWithMetadata from_libcudf(table_with_metadata& tbl_with_meta):
        """Create a Python TableWithMetadata from a libcudf table_with_metadata"""
        cdef TableWithMetadata out = TableWithMetadata.__new__(TableWithMetadata)
        out.tbl = Table.from_libcudf(move(tbl_with_meta.tbl))
        out.metadata = tbl_with_meta.metadata
        return out

cdef class SourceInfo:
    """A class containing details on a source to read from.

    For details, see :cpp:class:`cudf::io::source_info`.

    Parameters
    ----------
    sources : List[Union[str, os.PathLike, bytes, io.BytesIO]]
        A homogeneous list of sources (this can be a string filename,
        an os.PathLike, bytes, or an io.BytesIO) to read from.

        Mixing different types of sources will raise a `ValueError`.
    """

    def __init__(self, list sources):
        if not sources:
            raise ValueError("Need to pass at least one source")

        cdef vector[string] c_files

        if isinstance(sources[0], (os.PathLike, str)):
            c_files.reserve(len(sources))

            for src in sources:
                if not isinstance(src, (os.PathLike, str)):
                    raise ValueError("All sources must be of the same type!")
                if not os.path.isfile(src):
                    raise FileNotFoundError(errno.ENOENT,
                                            os.strerror(errno.ENOENT),
                                            src)

                c_files.push_back(<string> str(src).encode())

            self.c_obj = move(source_info(c_files))
            return

        # TODO: host_buffer is deprecated API, use host_span instead
        cdef vector[host_buffer] c_host_buffers
        cdef const unsigned char[::1] c_buffer
        cdef bint empty_buffer = False
        if isinstance(sources[0], bytes):
            empty_buffer = True
            for buffer in sources:
                if not isinstance(buffer, bytes):
                    raise ValueError("All sources must be of the same type!")
                if (len(buffer) > 0):
                    c_buffer = buffer
                    c_host_buffers.push_back(host_buffer(<char*>&c_buffer[0],
                                                         c_buffer.shape[0]))
                    empty_buffer = False
        elif isinstance(sources[0], io.BytesIO):
            for bio in sources:
                if not isinstance(bio, io.BytesIO):
                    raise ValueError("All sources must be of the same type!")
                c_buffer = bio.getbuffer()  # check if empty?
                c_host_buffers.push_back(host_buffer(<char*>&c_buffer[0],
                                                     c_buffer.shape[0]))

        self.c_obj = source_info(c_host_buffers)
