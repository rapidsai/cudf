# Copyright (c) 2024, NVIDIA CORPORATION.

from libcpp.string cimport string
from libcpp.utility cimport move
from libcpp.vector cimport vector
from pylibcudf.io.types cimport SourceInfo, TableWithMetadata
from pylibcudf.libcudf.io.avro cimport (
    avro_reader_options,
    read_avro as cpp_read_avro,
)
from pylibcudf.libcudf.types cimport size_type

__all__ = ["read_avro", "AvroReaderOptions", "AvroReaderOptionsBuilder"]


cdef class AvroReaderOptions:
    """
    The settings to use for ``read_avro``
    For details, see :cpp:class:`cudf::io::avro_reader_options`
    """
    @staticmethod
    def builder(SourceInfo source):
        """
        Create a AvroWriterOptionsBuilder object

        For details, see :cpp:func:`cudf::io::avro_reader_options::builder`

        Parameters
        ----------
        sink : SourceInfo
            The source to read the Avro file from.

        Returns
        -------
        AvroReaderOptionsBuilder
            Builder to build AvroReaderOptions
        """
        cdef AvroReaderOptionsBuilder avro_builder = AvroReaderOptionsBuilder.__new__(
            AvroReaderOptionsBuilder
        )
        avro_builder.c_obj = avro_reader_options.builder(source.c_obj)
        avro_builder.source = source
        return avro_builder

    cpdef void set_columns(self, list col_names):
        """
        Set names of the column to be read.

        Parameters
        ----------
        col_names : list[str]
            List of column names

        Returns
        -------
        None
        """
        cdef vector[string] vec
        vec.reserve(len(col_names))
        for name in col_names:
            vec.push_back(str(name).encode())
        self.c_obj.set_columns(vec)


cdef class AvroReaderOptionsBuilder:
    cpdef AvroReaderOptionsBuilder columns(self, list col_names):
        """
        Set names of the column to be read.

        Parameters
        ----------
        col_names : list
            List of column names

        Returns
        -------
        AvroReaderOptionsBuilder
        """
        cdef vector[string] vec
        vec.reserve(len(col_names))
        for name in col_names:
            vec.push_back(str(name).encode())
        self.c_obj.columns(vec)
        return self

    cpdef AvroReaderOptionsBuilder skip_rows(self, size_type skip_rows):
        """
        Sets number of rows to skip.

        Parameters
        ----------
        skip_rows : size_type
            Number of rows to skip from start

        Returns
        -------
        AvroReaderOptionsBuilder
        """
        self.c_obj.skip_rows(skip_rows)
        return self

    cpdef AvroReaderOptionsBuilder num_rows(self, size_type num_rows):
        """
        Sets number of rows to read.

        Parameters
        ----------
        num_rows : size_type
            Number of rows to read after skip

        Returns
        -------
        AvroReaderOptionsBuilder
        """
        self.c_obj.num_rows(num_rows)
        return self

    cpdef AvroReaderOptions build(self):
        """Create a AvroReaderOptions object"""
        cdef AvroReaderOptions avro_options = AvroReaderOptions.__new__(
            AvroReaderOptions
        )
        avro_options.c_obj = move(self.c_obj.build())
        avro_options.source = self.source
        return avro_options


cpdef TableWithMetadata read_avro(
    AvroReaderOptions options
):
    """
    Read from Avro format.

    The source to read from and options are encapsulated
    by the `options` object.

    For details, see :cpp:func:`read_avro`.

    Parameters
    ----------
    options: AvroReaderOptions
        Settings for controlling reading behavior
    """
    with nogil:
        c_result = move(cpp_read_avro(options.c_obj))

    return TableWithMetadata.from_libcudf(c_result)
