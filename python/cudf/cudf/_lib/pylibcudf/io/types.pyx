# Copyright (c) 2024, NVIDIA CORPORATION.

from cpython.buffer cimport PyBUF_READ
from cpython.memoryview cimport PyMemoryView_FromMemory
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libcpp.utility cimport move
from libcpp.vector cimport vector

from cudf._lib.pylibcudf.io.datasource cimport Datasource
from cudf._lib.pylibcudf.libcudf.io.data_sink cimport data_sink
from cudf._lib.pylibcudf.libcudf.io.datasource cimport datasource
from cudf._lib.pylibcudf.libcudf.io.types cimport (
    column_name_info,
    host_buffer,
    source_info,
    table_with_metadata,
)

import codecs
import errno
import io
import os


cdef class TableWithMetadata:
    """A container holding a table and its associated metadata
    (e.g. column names)

    For details, see :cpp:class:`cudf::io::table_with_metadata`.

    Parameters
    ----------
    tbl : Table
        The input table.
    column_names : list
        A list of tuples each containing the name of each column
        and the names of its child columns (in the same format).
        e.g.
        [("id", []), ("name", [("first", []), ("last", [])])]

    """
    def __init__(self, Table tbl, list column_names):
        self.tbl = tbl

        self.metadata.schema_info = self._make_column_info(column_names)

    cdef vector[column_name_info] _make_column_info(self, list column_names):
        cdef vector[column_name_info] col_name_infos
        cdef column_name_info info

        col_name_infos.reserve(len(column_names))

        for name, child_names in column_names:
            if not isinstance(name, str):
                raise ValueError("Column name must be a string!")

            info.name = <string> name.encode()
            info.children = self._make_column_info(child_names)

            col_name_infos.push_back(info)

        return col_name_infos

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
    sources : List[Union[str, os.PathLike, bytes, io.BytesIO, DataSource]]
        A homogeneous list of sources to read from.

        Mixing different types of sources will raise a `ValueError`.
    """

    def __init__(self, list sources):
        if not sources:
            raise ValueError("Need to pass at least one source")

        cdef vector[string] c_files
        cdef vector[datasource*] c_datasources

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
        elif isinstance(sources[0], Datasource):
            for csrc in sources:
                if not isinstance(csrc, Datasource):
                    raise ValueError("All sources must be of the same type!")
                c_datasources.push_back((<Datasource>csrc).get_datasource())
            self.c_obj = move(source_info(c_datasources))
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
        else:
            raise ValueError("Sources must be a list of str/paths, "
                             "bytes, io.BytesIO, or a Datasource")

        self.c_obj = source_info(c_host_buffers)


# Adapts a python io.IOBase object as a libcudf IO data_sink. This lets you
# write from cudf to any python file-like object (File/BytesIO/SocketIO etc)
cdef cppclass iobase_data_sink(data_sink):
    object buf

    iobase_data_sink(object buf_):
        this.buf = buf_

    void host_write(const void * data, size_t size) with gil:
        if isinstance(buf, io.TextIOBase):
            buf.write(PyMemoryView_FromMemory(<char*>data, size, PyBUF_READ)
                      .tobytes().decode())
        else:
            buf.write(PyMemoryView_FromMemory(<char*>data, size, PyBUF_READ))

    void flush() with gil:
        buf.flush()

    size_t bytes_written() with gil:
        return buf.tell()


cdef class SinkInfo:
    """A class containing details on a source to read from.

    For details, see :cpp:class:`cudf::io::sink_info`.

    Parameters
    ----------
    sinks : list of str, PathLike, BytesIO, StringIO

        A homogeneous list of sinks (this can be a string filename,
        bytes, or one of the Python I/O classes) to read from.

        Mixing different types of sinks will raise a `ValueError`.
    """

    def __init__(self, list sinks):
        cdef vector[data_sink *] data_sinks
        cdef vector[string] paths

        if not sinks:
            raise ValueError("Need to pass at least one sink")

        if isinstance(sinks[0], os.PathLike):
            sinks = [os.path.expanduser(s) for s in sinks]

        cdef object initial_sink_cls = type(sinks[0])

        if not all(isinstance(s, initial_sink_cls) for s in sinks):
            raise ValueError("All sinks must be of the same type!")

        if initial_sink_cls in {io.StringIO, io.BytesIO, io.TextIOBase}:
            data_sinks.reserve(len(sinks))
            if isinstance(sinks[0], (io.StringIO, io.BytesIO)):
                for s in sinks:
                    self.sink_storage.push_back(
                        unique_ptr[data_sink](new iobase_data_sink(s))
                    )
            elif isinstance(sinks[0], io.TextIOBase):
                for s in sinks:
                    if codecs.lookup(s).name not in ('utf-8', 'ascii'):
                        raise NotImplementedError(f"Unsupported encoding {s.encoding}")
                    self.sink_storage.push_back(
                        unique_ptr[data_sink](new iobase_data_sink(s.buffer))
                    )
            data_sinks.push_back(self.sink_storage.back().get())
        elif initial_sink_cls is str:
            paths.reserve(len(sinks))
            for s in sinks:
                paths.push_back(<string> s.encode())
        else:
            raise TypeError(
                "Unrecognized input type: {}".format(type(sinks[0]))
            )

        if data_sinks.size() > 0:
            self.c_obj = sink_info(data_sinks)
        else:
            # we don't have sinks so we must have paths to sinks
            self.c_obj = sink_info(paths)
