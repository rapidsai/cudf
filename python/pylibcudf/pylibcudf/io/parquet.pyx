# Copyright (c) 2024, NVIDIA CORPORATION.
from cython.operator cimport dereference
from libc.stdint cimport int64_t, uint8_t
from libcpp cimport bool
from libcpp.memory cimport unique_ptr, make_unique
from libcpp.string cimport string
from libcpp.utility cimport move
from libcpp.vector cimport vector
from pylibcudf.contiguous_split cimport HostBuffer
from pylibcudf.expressions cimport Expression
from pylibcudf.io.types cimport (
    SinkInfo,
    SourceInfo,
    PartitionInfo,
    TableInputMetadata,
    TableWithMetadata
)
from pylibcudf.libcudf.expressions cimport expression
from pylibcudf.libcudf.io.parquet cimport (
    chunked_parquet_reader as cpp_chunked_parquet_reader,
    parquet_reader_options,
    read_parquet as cpp_read_parquet,
    write_parquet as cpp_write_parquet,
    parquet_writer_options,
    parquet_chunked_writer as cpp_parquet_chunked_writer,
    chunked_parquet_writer_options,
    merge_row_group_metadata as cpp_merge_row_group_metadata,
)
from pylibcudf.libcudf.io.types cimport (
    compression_type,
    dictionary_policy as dictionary_policy_t,
    partition_info,
    statistics_freq,
    table_with_metadata,
)
from pylibcudf.libcudf.types cimport size_type
from pylibcudf.table cimport Table

__all__ = [
    "ChunkedParquetReader",
    "ParquetWriterOptions",
    "ParquetWriterOptionsBuilder",
    "read_parquet",
    "write_parquet",
    "ParquetReaderOptions",
    "ParquetReaderOptionsBuilder",
    "ChunkedParquetWriterOptions",
    "ChunkedParquetWriterOptionsBuilder"
    "merge_row_group_metadata",
]


cdef class ParquetReaderOptions:
    """The settings to use for ``read_parquet``
    For details, see :cpp:class:`cudf::io::parquet_reader_options`
    """
    @staticmethod
    def builder(SourceInfo source):
        """
        Create a ParquetReaderOptionsBuilder object

        For details, see :cpp:func:`cudf::io::parquet_reader_options::builder`

        Parameters
        ----------
        sink : SourceInfo
            The source to read the Parquet file from.

        Returns
        -------
        ParquetReaderOptionsBuilder
            Builder to build ParquetReaderOptions
        """
        cdef ParquetReaderOptionsBuilder parquet_builder = (
            ParquetReaderOptionsBuilder.__new__(ParquetReaderOptionsBuilder)
        )
        parquet_builder.c_obj = parquet_reader_options.builder(source.c_obj)
        parquet_builder.source = source
        return parquet_builder

    cpdef void set_row_groups(self, list row_groups):
        """
        Sets list of individual row groups to read.

        Parameters
        ----------
        row_groups : list
            List of row groups to read

        Returns
        -------
        None
        """
        cdef vector[vector[size_type]] outer
        cdef vector[size_type] inner
        for row_group in row_groups:
            for x in row_group:
                inner.push_back(x)
            outer.push_back(inner)
            inner.clear()

        self.c_obj.set_row_groups(outer)

    cpdef void set_num_rows(self, size_type nrows):
        """
        Sets number of rows to read.

        Parameters
        ----------
        nrows : size_type
            Number of rows to read after skip

        Returns
        -------
        None
        """
        self.c_obj.set_num_rows(nrows)

    cpdef void set_skip_rows(self, int64_t skip_rows):
        """
        Sets number of rows to skip.

        Parameters
        ----------
        skip_rows : int64_t
            Number of rows to skip from start

        Returns
        -------
        None
        """
        self.c_obj.set_skip_rows(skip_rows)

    cpdef void set_columns(self, list col_names):
        """
        Sets names of the columns to be read.

        Parameters
        ----------
        col_names : list
            List of column names

        Returns
        -------
        None
        """
        cdef vector[string] vec
        for name in col_names:
            vec.push_back(<string>str(name).encode())
        self.c_obj.set_columns(vec)

    cpdef void set_filter(self, Expression filter):
        """
        Sets AST based filter for predicate pushdown.

        Parameters
        ----------
        filter : Expression
            AST expression to use as filter

        Returns
        -------
        None
        """
        self.c_obj.set_filter(<expression &>dereference(filter.c_obj.get()))


cdef class ParquetReaderOptionsBuilder:
    cpdef ParquetReaderOptionsBuilder convert_strings_to_categories(self, bool val):
        """
        Sets enable/disable conversion of strings to categories.

        Parameters
        ----------
        val : bool
            Boolean value to enable/disable conversion of string columns to categories

        Returns
        -------
        ParquetReaderOptionsBuilder
        """
        self.c_obj.convert_strings_to_categories(val)
        return self

    cpdef ParquetReaderOptionsBuilder use_pandas_metadata(self, bool val):
        """
        Sets to enable/disable use of pandas metadata to read.

        Parameters
        ----------
        val : bool
            Boolean value whether to use pandas metadata

        Returns
        -------
        ParquetReaderOptionsBuilder
        """
        self.c_obj.use_pandas_metadata(val)
        return self

    cpdef ParquetReaderOptionsBuilder allow_mismatched_pq_schemas(self, bool val):
        """
        Sets to enable/disable reading of matching projected and filter
        columns from mismatched Parquet sources.

        Parameters
        ----------
        val : bool
            Boolean value whether to read matching projected and filter
            columns from mismatched Parquet sources.

        Returns
        -------
        ParquetReaderOptionsBuilder
        """
        self.c_obj.allow_mismatched_pq_schemas(val)
        return self

    cpdef ParquetReaderOptionsBuilder use_arrow_schema(self, bool val):
        """
        Sets to enable/disable use of arrow schema to read.

        Parameters
        ----------
        val : bool
            Boolean value whether to use arrow schema

        Returns
        -------
        ParquetReaderOptionsBuilder
        """
        self.c_obj.use_arrow_schema(val)
        return self

    cpdef build(self):
        """Create a ParquetReaderOptions object"""
        cdef ParquetReaderOptions parquet_options = ParquetReaderOptions.__new__(
            ParquetReaderOptions
        )
        parquet_options.c_obj = move(self.c_obj.build())
        parquet_options.source = self.source
        return parquet_options


cdef class ChunkedParquetReader:
    """
    Reads chunks of a Parquet file into a :py:class:`~.types.TableWithMetadata`.

    For details, see :cpp:class:`chunked_parquet_reader`.

    Parameters
    ----------
    options : ParquetReaderOptions
        Settings for controlling reading behavior
    chunk_read_limit : size_t, default 0
        Limit on total number of bytes to be returned per read,
        or 0 if there is no limit.
    pass_read_limit : size_t, default 1024000000
        Limit on the amount of memory used for reading and decompressing data
        or 0 if there is no limit.
    """
    def __init__(
        self,
        ParquetReaderOptions options,
        size_t chunk_read_limit=0,
        size_t pass_read_limit=1024000000,
    ):
        with nogil:
            self.reader.reset(
                new cpp_chunked_parquet_reader(
                    chunk_read_limit,
                    pass_read_limit,
                    options.c_obj,
                )
            )

    __hash__ = None

    cpdef bool has_next(self):
        """
        Returns True if there is another chunk in the Parquet file
        to be read.

        Returns
        -------
        True if we have not finished reading the file.
        """
        with nogil:
            return self.reader.get()[0].has_next()

    cpdef TableWithMetadata read_chunk(self):
        """
        Read the next chunk into a :py:class:`~.types.TableWithMetadata`

        Returns
        -------
        TableWithMetadata
            The Table and its corresponding metadata (column names) that were read in.
        """
        # Read Parquet
        cdef table_with_metadata c_result

        with nogil:
            c_result = move(self.reader.get()[0].read_chunk())

        return TableWithMetadata.from_libcudf(c_result)


cpdef read_parquet(ParquetReaderOptions options):
    """
    Read from Parquet format.

    The source to read from and options are encapsulated
    by the `options` object.

    For details, see :cpp:func:`read_parquet`.

    Parameters
    ----------
    options: ParquetReaderOptions
        Settings for controlling reading behavior
    """
    with nogil:
        c_result = move(cpp_read_parquet(options.c_obj))

    return TableWithMetadata.from_libcudf(c_result)


cdef class ParquetChunkedWriter:
    cpdef memoryview close(self, list metadata_file_path):
        """
        Closes the chunked Parquet writer.

        Parameters
        ----------
        metadata_file_path: list
            Column chunks file path to be set in the raw output metadata

        Returns
        -------
        None
        """
        cdef vector[string] column_chunks_file_paths
        cdef unique_ptr[vector[uint8_t]] out_metadata_c
        if metadata_file_path:
            for path in metadata_file_path:
                column_chunks_file_paths.push_back(path.encode())
        with nogil:
            out_metadata_c = move(self.c_obj.get()[0].close(column_chunks_file_paths))
        return memoryview(HostBuffer.from_unique_ptr(move(out_metadata_c)))

    cpdef void write(self, Table table, object partitions_info=None):
        """
        Writes table to output.

        Parameters
        ----------
        table: Table
            Table that needs to be written
        partitions_info: object, default None
            Optional partitions to divide the table into.
            If specified, must be same size as number of sinks.

        Returns
        -------
        None
        """
        if partitions_info is None:
            with nogil:
                self.c_obj.get()[0].write(table.view())
            return
        cdef vector[partition_info] partitions
        for part in partitions_info:
            partitions.push_back(
                partition_info(part[0], part[1])
            )
        with nogil:
            self.c_obj.get()[0].write(table.view(), partitions)

    @staticmethod
    def from_options(ChunkedParquetWriterOptions options):
        """
        Creates a chunked Parquet writer from options

        Parameters
        ----------
        options: ChunkedParquetWriterOptions
            Settings for controlling writing behavior

        Returns
        -------
        ParquetChunkedWriter
        """
        cdef ParquetChunkedWriter parquet_writer = ParquetChunkedWriter.__new__(
            ParquetChunkedWriter
        )
        parquet_writer.c_obj.reset(new cpp_parquet_chunked_writer(options.c_obj))
        return parquet_writer


cdef class ChunkedParquetWriterOptions:
    @staticmethod
    def builder(SinkInfo sink):
        """
        Create builder to create ChunkedParquetWriterOptions.

        Parameters
        ----------
        sink: SinkInfo
            The sink used for writer output

        Returns
        -------
        ChunkedParquetWriterOptionsBuilder
        """
        cdef ChunkedParquetWriterOptionsBuilder parquet_builder = (
            ChunkedParquetWriterOptionsBuilder.__new__(
                ChunkedParquetWriterOptionsBuilder
            )
        )
        parquet_builder.c_obj = chunked_parquet_writer_options.builder(sink.c_obj)
        parquet_builder.sink = sink
        return parquet_builder

    cpdef void set_dictionary_policy(self, dictionary_policy_t policy):
        """
        Sets the policy for dictionary use.

        Parameters
        ----------
        policy : DictionaryPolicy
            Policy for dictionary use

        Returns
        -------
        None
        """
        self.c_obj.set_dictionary_policy(policy)


cdef class ChunkedParquetWriterOptionsBuilder:
    cpdef ChunkedParquetWriterOptionsBuilder metadata(
        self,
        TableInputMetadata metadata
    ):
        self.c_obj.metadata(metadata.c_obj)
        return self

    cpdef ChunkedParquetWriterOptionsBuilder key_value_metadata(self, list metadata):
        """
        Sets Key-Value footer metadata.

        Parameters
        ----------
        metadata : list[dict[str, str]]
            Key-Value footer metadata

        Returns
        -------
        Self
        """
        self.c_obj.key_value_metadata(
            [
                {key.encode(): value.encode() for key, value in mapping.items()}
                for mapping in metadata
            ]
        )
        return self

    cpdef ChunkedParquetWriterOptionsBuilder compression(
        self,
        compression_type compression
    ):
        """
        Sets compression type.

        Parameters
        ----------
        compression : CompressionType
            The compression type to use

        Returns
        -------
        Self
        """
        self.c_obj.compression(compression)
        return self

    cpdef ChunkedParquetWriterOptionsBuilder stats_level(self, statistics_freq sf):
        """
        Sets the level of statistics.

        Parameters
        ----------
        sf : StatisticsFreq
            Level of statistics requested in the output file

        Returns
        -------
        Self
        """
        self.c_obj.stats_level(sf)
        return self

    cpdef ChunkedParquetWriterOptionsBuilder row_group_size_bytes(self, size_t val):
        """
        Sets the maximum row group size, in bytes.

        Parameters
        ----------
        val : size_t
            Maximum row group size, in bytes to set

        Returns
        -------
        Self
        """
        self.c_obj.row_group_size_bytes(val)
        return self

    cpdef ChunkedParquetWriterOptionsBuilder row_group_size_rows(self, size_type val):
        """
        Sets the maximum row group size, in rows.

        Parameters
        ----------
        val : size_type
            Maximum row group size, in rows to set

        Returns
        -------
        Self
        """
        self.c_obj.row_group_size_rows(val)
        return self

    cpdef ChunkedParquetWriterOptionsBuilder max_page_size_bytes(self, size_t val):
        """
        Sets the maximum uncompressed page size, in bytes.

        Parameters
        ----------
        val : size_t
            Maximum uncompressed page size, in bytes to set

        Returns
        -------
        Self
        """
        self.c_obj.max_page_size_bytes(val)
        return self

    cpdef ChunkedParquetWriterOptionsBuilder max_page_size_rows(self, size_type val):
        """
        Sets the maximum page size, in rows.

        Parameters
        ----------
        val : size_type
            Maximum page size, in rows to set.

        Returns
        -------
        Self
        """
        self.c_obj.max_page_size_rows(val)
        return self

    cpdef ChunkedParquetWriterOptionsBuilder max_dictionary_size(self, size_t val):
        """
        Sets the maximum dictionary size, in bytes.

        Parameters
        ----------
        val : size_t
            Sets the maximum dictionary size, in bytes.

        Returns
        -------
        Self
        """
        self.c_obj.max_dictionary_size(val)
        return self

    cpdef ChunkedParquetWriterOptionsBuilder write_arrow_schema(self, bool enabled):
        """
        Set to true if arrow schema is to be written.

        Parameters
        ----------
        enabled : bool
            Boolean value to enable/disable writing of arrow schema.

        Returns
        -------
        Self
        """
        self.c_obj.write_arrow_schema(enabled)
        return self

    cpdef ChunkedParquetWriterOptions build(self):
        """Create a ChunkedParquetWriterOptions object"""
        cdef ChunkedParquetWriterOptions parquet_options = (
            ChunkedParquetWriterOptions.__new__(ChunkedParquetWriterOptions)
        )
        parquet_options.c_obj = move(self.c_obj.build())
        parquet_options.sink = self.sink
        return parquet_options


cdef class ParquetWriterOptions:

    @staticmethod
    def builder(SinkInfo sink, Table table):
        """
        Create builder to create ParquetWriterOptionsBuilder.

        Parameters
        ----------
        sink : SinkInfo
            The sink used for writer output

        table : Table
            Table to be written to output

        Returns
        -------
        ParquetWriterOptionsBuilder
        """
        cdef ParquetWriterOptionsBuilder bldr = ParquetWriterOptionsBuilder.__new__(
            ParquetWriterOptionsBuilder
        )
        bldr.c_obj = parquet_writer_options.builder(sink.c_obj, table.view())
        bldr.table_ref = table
        bldr.sink_ref = sink
        return bldr

    cpdef void set_partitions(self, list partitions):
        """
        Sets partitions.

        Parameters
        ----------
        partitions : list[Partitions]
            Partitions of input table in {start_row, num_rows} pairs.

        Returns
        -------
        None
        """
        cdef vector[partition_info] c_partions
        cdef PartitionInfo partition

        c_partions.reserve(len(partitions))
        for partition in partitions:
            c_partions.push_back(partition.c_obj)

        self.c_obj.set_partitions(c_partions)

    cpdef void set_column_chunks_file_paths(self, list file_paths):
        """
        Sets column chunks file path to be set in the raw output metadata.

        Parameters
        ----------
        file_paths : list[str]
            Vector of strings which indicate file paths.

        Returns
        -------
        None
        """
        self.c_obj.set_column_chunks_file_paths([fp.encode() for fp in file_paths])

    cpdef void set_row_group_size_bytes(self, size_t size_bytes):
        """
        Sets the maximum row group size, in bytes.

        Parameters
        ----------
        size_bytes : int
            Maximum row group size, in bytes to set

        Returns
        -------
        None
        """
        self.c_obj.set_row_group_size_bytes(size_bytes)

    cpdef void set_row_group_size_rows(self, size_type size_rows):
        """
        Sets the maximum row group size, in rows.

        Parameters
        ----------
        size_rows : int
            Maximum row group size, in rows to set

        Returns
        -------
        None
        """
        self.c_obj.set_row_group_size_rows(size_rows)

    cpdef void set_max_page_size_bytes(self, size_t size_bytes):
        """
        Sets the maximum uncompressed page size, in bytes.

        Parameters
        ----------
        size_bytes : int
            Maximum uncompressed page size, in bytes to set

        Returns
        -------
        None
        """
        self.c_obj.set_max_page_size_bytes(size_bytes)

    cpdef void set_max_page_size_rows(self, size_type size_rows):
        """
        Sets the maximum page size, in rows.

        Parameters
        ----------
        size_rows : int
            Maximum page size, in rows to set.

        Returns
        -------
        None
        """
        self.c_obj.set_max_page_size_rows(size_rows)

    cpdef void set_max_dictionary_size(self, size_t size_bytes):
        """
        Sets the maximum dictionary size, in bytes.

        Parameters
        ----------
        size_bytes : int
            Sets the maximum dictionary size, in bytes.

        Returns
        -------
        None
        """
        self.c_obj.set_max_dictionary_size(size_bytes)


cdef class ParquetWriterOptionsBuilder:

    cpdef ParquetWriterOptionsBuilder metadata(self, TableInputMetadata metadata):
        """
        Sets metadata.

        Parameters
        ----------
        metadata : TableInputMetadata
            Associated metadata

        Returns
        -------
        Self
        """
        self.c_obj.metadata(metadata.c_obj)
        return self

    cpdef ParquetWriterOptionsBuilder key_value_metadata(self, list metadata):
        """
        Sets Key-Value footer metadata.

        Parameters
        ----------
        metadata : list[dict[str, str]]
            Key-Value footer metadata

        Returns
        -------
        Self
        """
        self.c_obj.key_value_metadata(
            [
                {key.encode(): value.encode() for key, value in mapping.items()}
                for mapping in metadata
            ]
        )
        return self

    cpdef ParquetWriterOptionsBuilder compression(self, compression_type compression):
        """
        Sets compression type.

        Parameters
        ----------
        compression : CompressionType
            The compression type to use

        Returns
        -------
        Self
        """
        self.c_obj.compression(compression)
        return self

    cpdef ParquetWriterOptionsBuilder stats_level(self, statistics_freq sf):
        """
        Sets the level of statistics.

        Parameters
        ----------
        sf : StatisticsFreq
            Level of statistics requested in the output file

        Returns
        -------
        Self
        """
        self.c_obj.stats_level(sf)
        return self

    cpdef ParquetWriterOptionsBuilder int96_timestamps(self, bool enabled):
        """
        Sets whether timestamps are written as int96 or timestamp micros.

        Parameters
        ----------
        enabled : bool
            Boolean value to enable/disable int96 timestamps

        Returns
        -------
        Self
        """
        self.c_obj.int96_timestamps(enabled)
        return self

    cpdef ParquetWriterOptionsBuilder write_v2_headers(self, bool enabled):
        """
        Set to true to write V2 page headers, otherwise false to write V1 page headers.

        Parameters
        ----------
        enabled : bool
            Boolean value to enable/disable writing of V2 page headers.

        Returns
        -------
        Self
        """
        self.c_obj.write_v2_headers(enabled)
        return self

    cpdef ParquetWriterOptionsBuilder dictionary_policy(self, dictionary_policy_t val):
        """
        Sets the policy for dictionary use.

        Parameters
        ----------
        val : DictionaryPolicy
            Policy for dictionary use.

        Returns
        -------
        Self
        """
        self.c_obj.dictionary_policy(val)
        return self

    cpdef ParquetWriterOptionsBuilder utc_timestamps(self, bool enabled):
        """
        Set to true if timestamps are to be written as UTC.

        Parameters
        ----------
        enabled : bool
            Boolean value to enable/disable writing of timestamps as UTC.

        Returns
        -------
        Self
        """
        self.c_obj.utc_timestamps(enabled)
        return self

    cpdef ParquetWriterOptionsBuilder write_arrow_schema(self, bool enabled):
        """
        Set to true if arrow schema is to be written.

        Parameters
        ----------
        enabled : bool
            Boolean value to enable/disable writing of arrow schema.

        Returns
        -------
        Self
        """
        self.c_obj.write_arrow_schema(enabled)
        return self

    cpdef ParquetWriterOptions build(self):
        """
        Create a ParquetWriterOptions from the set options.

        Returns
        -------
        ParquetWriterOptions
        """
        cdef ParquetWriterOptions parquet_options = ParquetWriterOptions.__new__(
            ParquetWriterOptions
        )
        parquet_options.c_obj = move(self.c_obj.build())
        parquet_options.table_ref = self.table_ref
        parquet_options.sink_ref = self.sink_ref
        return parquet_options


cpdef memoryview write_parquet(ParquetWriterOptions options):
    """
    Writes a set of columns to parquet format.

    Parameters
    ----------
    options : ParquetWriterOptions
        Settings for controlling writing behavior

    Returns
    -------
    memoryview
        A blob that contains the file metadata
        (parquet FileMetadata thrift message) if requested in
        parquet_writer_options (empty blob otherwise).
    """
    cdef unique_ptr[vector[uint8_t]] c_result

    with nogil:
        c_result = cpp_write_parquet(move(options.c_obj))

    return memoryview(HostBuffer.from_unique_ptr(move(c_result)))


cpdef memoryview merge_row_group_metadata(list metdata_list):
    """
    Merges multiple raw metadata blobs that were previously
    created by write_parquet into a single metadata blob.

    For details, see :cpp:func:`merge_row_group_metadata`.

    Parameters
    ----------
    metdata_list : list
        List of input file metadata

    Returns
    -------
    memoryview
        A parquet-compatible blob that contains the data for all row groups in the list
    """
    cdef vector[unique_ptr[vector[uint8_t]]] list_c
    cdef unique_ptr[vector[uint8_t]] output_c

    for blob in metdata_list:
        list_c.push_back(move(make_unique[vector[uint8_t]](<vector[uint8_t]> blob)))

    with nogil:
        output_c = move(cpp_merge_row_group_metadata(list_c))

    return memoryview(HostBuffer.from_unique_ptr(move(output_c)))
