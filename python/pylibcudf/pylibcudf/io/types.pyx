# Copyright (c) 2024, NVIDIA CORPORATION.

from cpython.buffer cimport PyBUF_READ
from cpython.memoryview cimport PyMemoryView_FromMemory
from libc.stdint cimport uint8_t, int32_t
from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libcpp.utility cimport move
from libcpp.vector cimport vector
from pylibcudf.io.datasource cimport Datasource
from pylibcudf.libcudf.io.data_sink cimport data_sink
from pylibcudf.libcudf.io.datasource cimport datasource
from pylibcudf.libcudf.io.types cimport (
    column_encoding,
    column_in_metadata,
    column_name_info,
    host_buffer,
    partition_info,
    source_info,
    table_input_metadata,
    table_with_metadata,
)
from pylibcudf.libcudf.types cimport size_type

import codecs
import errno
import io
import os
import re

from pylibcudf.libcudf.io.json import \
    json_recovery_mode_t as JSONRecoveryMode  # no-cython-lint
from pylibcudf.libcudf.io.types import (
    compression_type as CompressionType,  # no-cython-lint
    column_encoding as ColumnEncoding,  # no-cython-lint
    dictionary_policy as DictionaryPolicy,  # no-cython-lint
    quote_style as QuoteStyle,  # no-cython-lint
    statistics_freq as StatisticsFreq, # no-cython-lint
)

__all__ = [
    "ColumnEncoding",
    "CompressionType",
    "DictionaryPolicy",
    "JSONRecoveryMode",
    "PartitionInfo",
    "QuoteStyle",
    "SinkInfo",
    "SourceInfo",
    "StatisticsFreq",
    "TableInputMetadata",
    "TableWithMetadata",
]

cdef class PartitionInfo:
    """
    Information used while writing partitioned datasets.

    Parameters
    ----------
    start_row : int
        The start row of the partition.

    num_rows : int
        The number of rows in the partition.
    """
    def __init__(self, size_type start_row, size_type num_rows):
        self.c_obj = partition_info(start_row, num_rows)


cdef class ColumnInMetadata:
    """
    Metadata for a column
    """

    @staticmethod
    cdef ColumnInMetadata from_metadata(column_in_metadata metadata):
        """
        Construct a ColumnInMetadata.

        Parameters
        ----------
        metadata : column_in_metadata
        """
        cdef ColumnInMetadata col_metadata = ColumnInMetadata.__new__(ColumnInMetadata)
        col_metadata.c_obj = metadata
        return col_metadata

    cpdef ColumnInMetadata set_name(self, str name):
        """
        Set the name of this column.

        Parameters
        ----------
        name : str
            Name of the column

        Returns
        -------
        Self
        """
        self.c_obj.set_name(name.encode())
        return self

    cpdef ColumnInMetadata set_nullability(self, bool nullable):
        """
        Set the nullability of this column.

        Parameters
        ----------
        nullable : bool
            Whether this column is nullable

        Returns
        -------
        Self
        """
        self.c_obj.set_nullability(nullable)
        return self

    cpdef ColumnInMetadata set_list_column_as_map(self):
        """
        Specify that this list column should be encoded as a map in the
        written file.

        Returns
        -------
        Self
        """
        self.c_obj.set_list_column_as_map()
        return self

    cpdef ColumnInMetadata set_int96_timestamps(self, bool req):
        """
        Specifies whether this timestamp column should be encoded using
        the deprecated int96.

        Parameters
        ----------
        req : bool
            True = use int96 physical type. False = use int64 physical type.

        Returns
        -------
        Self
        """
        self.c_obj.set_int96_timestamps(req)
        return self

    cpdef ColumnInMetadata set_decimal_precision(self, uint8_t precision):
        """
        Set the decimal precision of this column.
        Only valid if this column is a decimal (fixed-point) type.

        Parameters
        ----------
        precision : int
            The integer precision to set for this decimal column

        Returns
        -------
        Self
        """
        self.c_obj.set_decimal_precision(precision)
        return self

    cpdef ColumnInMetadata child(self, size_type i):
        """
        Get reference to a child of this column.

        Parameters
        ----------
        i : int
            Index of the child to get.

        Returns
        -------
        ColumnInMetadata
        """
        return ColumnInMetadata.from_metadata(self.c_obj.child(i))

    cpdef ColumnInMetadata set_output_as_binary(self, bool binary):
        """
        Specifies whether this column should be written as binary or string data.

        Parameters
        ----------
        binary : bool
            True = use binary data type. False = use string data type

        Returns
        -------
        Self
        """
        self.c_obj.set_output_as_binary(binary)
        return self

    cpdef ColumnInMetadata set_type_length(self, int32_t type_length):
        """
        Sets the length of fixed length data.

        Parameters
        ----------
        type_length : int
            Size of the data type in bytes

        Returns
        -------
        Self
        """
        self.c_obj.set_type_length(type_length)
        return self

    cpdef ColumnInMetadata set_skip_compression(self, bool skip):
        """
        Specifies whether this column should not be compressed
        regardless of the compression.

        Parameters
        ----------
        skip : bool
            If `true` do not compress this column

        Returns
        -------
        Self
        """
        self.c_obj.set_skip_compression(skip)
        return self

    cpdef ColumnInMetadata set_encoding(self, column_encoding encoding):
        """
        Specifies whether this column should not be compressed
        regardless of the compression.

        Parameters
        ----------
        encoding : ColumnEncoding
            The encoding to use

        Returns
        -------
        ColumnInMetadata
        """
        self.c_obj.set_encoding(encoding)
        return self

    cpdef str get_name(self):
        """
        Get the name of this column.

        Returns
        -------
        str
            The name of this column
        """
        return self.c_obj.get_name().decode()


cdef class TableInputMetadata:
    """
    Metadata for a table

    Parameters
    ----------
    table : Table
        The Table to construct metadata for
    """
    def __init__(self, Table table):
        self.c_obj = table_input_metadata(table.view())


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

    __hash__ = None

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

    cdef list _make_columns_list(self, dict child_dict):
        cdef list names = []
        for child in child_dict:
            grandchildren = self._make_columns_list(child_dict[child])
            names.append((child, grandchildren))
        return names

    def column_names(self, include_children=False):
        """
        Return a list containing the column names of the table
        """
        cdef list names = []
        cdef str name
        cdef dict child_names = self.child_names
        for col_info in self.metadata.schema_info:
            name = col_info.name.decode()
            if include_children:
                children = self._make_columns_list(child_names[name])
                names.append((name, children))
            else:
                names.append(name)
        return names

    @property
    def child_names(self):
        """
        Return a dictionary mapping the names of columns with children
        to the names of their child columns
        """
        return TableWithMetadata._parse_col_names(self.metadata.schema_info)

    @staticmethod
    cdef dict _parse_col_names(vector[column_name_info] infos):
        cdef dict child_names = dict()
        cdef dict names = dict()
        for col_info in infos:
            child_names = TableWithMetadata._parse_col_names(col_info.children)
            names[col_info.name.decode()] = child_names
        return names

    @staticmethod
    cdef TableWithMetadata from_libcudf(table_with_metadata& tbl_with_meta):
        """Create a Python TableWithMetadata from a libcudf table_with_metadata"""
        cdef TableWithMetadata out = TableWithMetadata.__new__(TableWithMetadata)
        out.tbl = Table.from_libcudf(move(tbl_with_meta.tbl))
        out.metadata = tbl_with_meta.metadata
        return out

    @property
    def per_file_user_data(self):
        """
        Returns a list containing a dict
        containing file-format specific metadata,
        for each file being read in.
        """
        return self.metadata.per_file_user_data


cdef class SourceInfo:
    """A class containing details on a source to read from.

    For details, see :cpp:class:`cudf::io::source_info`.

    Parameters
    ----------
    sources : List[Union[str, os.PathLike, bytes, io.BytesIO, DataSource]]
        A homogeneous list of sources to read from.

        Mixing different types of sources will raise a `ValueError`.
    """
    # Regular expression that match remote file paths supported by libcudf
    _is_remote_file_pattern = re.compile(r"^s3://", re.IGNORECASE)

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
                if not (os.path.isfile(src) or self._is_remote_file_pattern.match(src)):
                    raise FileNotFoundError(
                        errno.ENOENT, os.strerror(errno.ENOENT), src
                    )
                # TODO: Keep the sources alive (self.byte_sources = sources)
                # for str data (e.g. read_json)?
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
        cdef list new_sources = []

        if isinstance(sources[0], io.StringIO):
            for buffer in sources:
                if not isinstance(buffer, io.StringIO):
                    raise ValueError("All sources must be of the same type!")
                new_sources.append(buffer.read().encode())
            sources = new_sources
            self.byte_sources = sources
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
                             "bytes, io.BytesIO, io.StringIO, or a Datasource")

        if empty_buffer is True:
            c_host_buffers.push_back(host_buffer(<char*>NULL, 0))

        self.c_obj = source_info(c_host_buffers)

    __hash__ = None


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
    """
    A class containing details about destinations (sinks) to write data to.

    For more details, see :cpp:class:`cudf::io::sink_info`.

    Parameters
    ----------
    sinks : list of str, PathLike, or io.IOBase instances
        A list of sinks to write data to. Each sink can be:

        - A string representing a filename.
        - A PathLike object.
        - An instance of a Python I/O class that is a subclass of io.IOBase
          (eg., io.BytesIO, io.StringIO).

        The list must be homogeneous in type unless all sinks are instances
        of subclasses of io.IOBase. Mixing different types of sinks
        (that are not all io.IOBase instances) will raise a ValueError.
    """

    def __init__(self, list sinks):
        cdef vector[data_sink *] data_sinks
        cdef vector[string] paths

        if not sinks:
            raise ValueError("At least one sink must be provided.")

        if isinstance(sinks[0], os.PathLike):
            sinks = [os.path.expanduser(s) for s in sinks]

        cdef object initial_sink_cls = type(sinks[0])

        if not all(
            isinstance(s, initial_sink_cls) or (
                isinstance(sinks[0], io.IOBase) and isinstance(s, io.IOBase)
            ) for s in sinks
        ):
            raise ValueError(
                "All sinks must be of the same type unless they are all instances "
                "of subclasses of io.IOBase."
            )

        if isinstance(sinks[0], io.IOBase):
            data_sinks.reserve(len(sinks))
            for s in sinks:
                if isinstance(s, (io.StringIO, io.BytesIO)):
                    self.sink_storage.push_back(
                        unique_ptr[data_sink](new iobase_data_sink(s))
                    )
                elif isinstance(s, io.TextIOBase):
                    if codecs.lookup(s.encoding).name not in ('utf-8', 'ascii'):
                        raise NotImplementedError(f"Unsupported encoding {s.encoding}")
                    self.sink_storage.push_back(
                        unique_ptr[data_sink](new iobase_data_sink(s.buffer))
                    )
                else:
                    self.sink_storage.push_back(
                        unique_ptr[data_sink](new iobase_data_sink(s))
                    )
                data_sinks.push_back(self.sink_storage.back().get())
        elif isinstance(sinks[0], str):
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

    __hash__ = None
