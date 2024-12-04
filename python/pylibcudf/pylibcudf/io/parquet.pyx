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
    "merge_row_group_metadata",
]

cdef parquet_reader_options _setup_parquet_reader_options(
    SourceInfo source_info,
    list columns = None,
    list row_groups = None,
    Expression filters = None,
    bool convert_strings_to_categories = False,
    bool use_pandas_metadata = True,
    int64_t skip_rows = 0,
    size_type nrows = -1,
    bool allow_mismatched_pq_schemas=False,
    # ReaderColumnSchema reader_column_schema = None,
    # DataType timestamp_type = DataType(type_id.EMPTY)
):
    cdef vector[string] col_vec
    cdef parquet_reader_options opts = (
        parquet_reader_options.builder(source_info.c_obj)
        .convert_strings_to_categories(convert_strings_to_categories)
        .use_pandas_metadata(use_pandas_metadata)
        .allow_mismatched_pq_schemas(allow_mismatched_pq_schemas)
        .use_arrow_schema(True)
        .build()
    )
    if row_groups is not None:
        opts.set_row_groups(row_groups)
    if nrows != -1:
        opts.set_num_rows(nrows)
    if skip_rows != 0:
        opts.set_skip_rows(skip_rows)
    if columns is not None:
        col_vec.reserve(len(columns))
        for col in columns:
            col_vec.push_back(<string>str(col).encode())
        opts.set_columns(col_vec)
    if filters is not None:
        opts.set_filter(<expression &>dereference(filters.c_obj.get()))
    return opts


cdef class ChunkedParquetReader:
    """
    Reads chunks of a Parquet file into a :py:class:`~.types.TableWithMetadata`.

    For details, see :cpp:class:`chunked_parquet_reader`.

    Parameters
    ----------
    source_info : SourceInfo
        The SourceInfo object to read the Parquet file from.
    columns : list, default None
        The names of the columns to be read
    row_groups : list[list[size_type]], default None
        List of row groups to be read.
    use_pandas_metadata : bool, default True
        If True, return metadata about the index column in
        the per-file user metadata of the ``TableWithMetadata``
    convert_strings_to_categories : bool, default False
        Whether to convert string columns to the category type
    skip_rows : int64_t, default 0
        The number of rows to skip from the start of the file.
    nrows : size_type, default -1
        The number of rows to read. By default, read the entire file.
    chunk_read_limit : size_t, default 0
        Limit on total number of bytes to be returned per read,
        or 0 if there is no limit.
    pass_read_limit : size_t, default 1024000000
        Limit on the amount of memory used for reading and decompressing data
        or 0 if there is no limit.
    allow_mismatched_pq_schemas : bool, default False
        Whether to read (matching) columns specified in `columns` from
        the input files with otherwise mismatched schemas.
    """
    def __init__(
        self,
        SourceInfo source_info,
        list columns=None,
        list row_groups=None,
        bool use_pandas_metadata=True,
        bool convert_strings_to_categories=False,
        int64_t skip_rows = 0,
        size_type nrows = -1,
        size_t chunk_read_limit=0,
        size_t pass_read_limit=1024000000,
        bool allow_mismatched_pq_schemas=False
    ):

        cdef parquet_reader_options opts = _setup_parquet_reader_options(
            source_info,
            columns,
            row_groups,
            filters=None,
            convert_strings_to_categories=convert_strings_to_categories,
            use_pandas_metadata=use_pandas_metadata,
            skip_rows=skip_rows,
            nrows=nrows,
            allow_mismatched_pq_schemas=allow_mismatched_pq_schemas,
        )

        with nogil:
            self.reader.reset(
                new cpp_chunked_parquet_reader(
                    chunk_read_limit,
                    pass_read_limit,
                    opts
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

cpdef read_parquet(
    SourceInfo source_info,
    list columns = None,
    list row_groups = None,
    Expression filters = None,
    bool convert_strings_to_categories = False,
    bool use_pandas_metadata = True,
    int64_t skip_rows = 0,
    size_type nrows = -1,
    bool allow_mismatched_pq_schemas = False,
    # Disabled, these aren't used by cudf-python
    # we should only add them back in if there's user demand
    # ReaderColumnSchema reader_column_schema = None,
    # DataType timestamp_type = DataType(type_id.EMPTY)
):
    """Reads an Parquet file into a :py:class:`~.types.TableWithMetadata`.

    For details, see :cpp:func:`read_parquet`.

    Parameters
    ----------
    source_info : SourceInfo
        The SourceInfo object to read the Parquet file from.
    columns : list, default None
        The string names of the columns to be read.
    row_groups : list[list[size_type]], default None
        List of row groups to be read.
    filters : Expression, default None
        An AST :py:class:`pylibcudf.expressions.Expression`
        to use for predicate pushdown.
    convert_strings_to_categories : bool, default False
        Whether to convert string columns to the category type
    use_pandas_metadata : bool, default True
        If True, return metadata about the index column in
        the per-file user metadata of the ``TableWithMetadata``
    skip_rows : int64_t, default 0
        The number of rows to skip from the start of the file.
    nrows : size_type, default -1
        The number of rows to read. By default, read the entire file.
    allow_mismatched_pq_schemas : bool, default False
        If True, enable reading (matching) columns specified in `columns`
        from the input files with otherwise mismatched schemas.

    Returns
    -------
    TableWithMetadata
        The Table and its corresponding metadata (column names) that were read in.
    """
    cdef table_with_metadata c_result
    cdef parquet_reader_options opts = _setup_parquet_reader_options(
        source_info,
        columns,
        row_groups,
        filters,
        convert_strings_to_categories,
        use_pandas_metadata,
        skip_rows,
        nrows,
        allow_mismatched_pq_schemas,
    )

    with nogil:
        c_result = move(cpp_read_parquet(opts))

    return TableWithMetadata.from_libcudf(c_result)


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
    cdef parquet_writer_options c_options = options.c_obj
    cdef unique_ptr[vector[uint8_t]] c_result

    with nogil:
        c_result = cpp_write_parquet(c_options)

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
