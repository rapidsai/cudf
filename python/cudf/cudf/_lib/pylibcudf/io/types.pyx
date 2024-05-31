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

    @property
    def columns(self):
        return self.tbl._columns

    @property
    def column_names(self):
        # TODO: Handle nesting (columns with child columns)
        return [col_info.name.decode() for col_info in self.metadata.schema_info]

    @staticmethod
    cdef TableWithMetadata from_libcudf(table_with_metadata& tbl_with_meta):
        """Create a Python TableWithMetadata from a libcudf table_with_metadata"""
        cdef TableWithMetadata out = TableWithMetadata.__new__(TableWithMetadata)
        out.tbl = Table.from_libcudf(move(tbl_with_meta.tbl))
        out.metadata = tbl_with_meta.metadata
        return out

cdef class SourceInfo:

    def __init__(self, list sources):
        if not sources:
            raise ValueError("Need to pass at least one source")

        if isinstance(sources[0], os.PathLike) or isinstance(sources[0], str):
            sources = [str(src) for src in sources]

        cdef vector[string] c_files
        if isinstance(sources[0], str):
            # If source is a file, return source_info where type=FILEPATH
            if not all(os.path.isfile(file) for file in sources):
                raise FileNotFoundError(errno.ENOENT,
                                        os.strerror(errno.ENOENT),
                                        sources)

            c_files.reserve(len(sources))
            for src in sources:
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
                if (len(buffer) > 0):
                    c_buffer = buffer
                    c_host_buffers.push_back(host_buffer(<char*>&c_buffer[0],
                                                         c_buffer.shape[0]))
                    empty_buffer = False
        elif isinstance(sources[0], io.BytesIO):
            for bio in sources:
                c_buffer = bio.getbuffer()  # check if empty?
                c_host_buffers.push_back(host_buffer(<char*>&c_buffer[0],
                                                     c_buffer.shape[0]))

        self.c_obj = source_info(c_host_buffers)
