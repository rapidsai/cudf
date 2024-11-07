# Copyright (c) 2024, NVIDIA CORPORATION.
from cython.operator cimport dereference
from libc.stdint cimport int64_t, uint8_t
from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libcpp.utility cimport move
from libcpp.vector cimport vector
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
    cdef ParquetWriterOptionsBuilder builder(SinkInfo sink, Table table):
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
        bldr.builder = parquet_writer_options.builder(sink.c_obj, table.view())
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

        self.options.set_partitions(c_partions)

    cpdef void set_column_chunks_file_paths(self, list file_paths):
        """
        Sets column chunks file path to be set in the raw output metadata.

        Parameters
        ----------
        file_paths : list[str]
            Vector of Strings which indicates file path.

        Returns
        -------
        None
        """
        self.options.set_column_chunks_file_paths([fp.encode() for fp in file_paths])

    cpdef void set_row_group_size_bytes(self, int size_bytes):
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
        self.options.set_row_group_size_bytes(size_bytes)

    cpdef void set_row_group_size_rows(self, int size_rows):
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
        self.options.set_row_group_size_rows(size_rows)

    cpdef void set_max_page_size_bytes(self, int size_bytes):
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
        self.options.set_max_page_size_bytes(size_bytes)

    cpdef void set_max_page_size_rows(self, int size_rows):
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
        self.options.set_max_page_size_rows(size_rows)

    cpdef void set_max_dictionary_size(self, int size_rows):
        """
        Sets the maximum dictionary size, in bytes.

        Parameters
        ----------
        size_rows : int
            Sets the maximum dictionary size, in bytes..

        Returns
        -------
        None
        """
        self.options.set_max_dictionary_size(size_rows)


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
        self.builder.metadata(metadata.c_obj)
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
        self.builder.key_value_metadata(
            [
                {key.encode(): value.encode() for key, value in mapping.items()}
                for mapping in metadata
            ]
        )
        return self

    cpdef ParquetWriterOptionsBuilder compression(self, compression_type compression):
        """
        Sets Key-Value footer metadata.

        Parameters
        ----------
        compression : CompressionType
            The compression type to use

        Returns
        -------
        Self
        """
        self.builder.compression(compression)
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
        self.builder.stats_level(sf)
        return self

    cpdef ParquetWriterOptionsBuilder int96_timestamps(self, bool enabled):
        """
        Sets whether int96 timestamps are written or not.

        Parameters
        ----------
        enabled : bool
            Boolean value to enable/disable int96 timestamps

        Returns
        -------
        Self
        """
        self.builder.int96_timestamps(enabled)
        return self

    cpdef ParquetWriterOptionsBuilder write_v2_headers(self, bool enabled):
        """
        Set to true if V2 page headers are to be written.

        Parameters
        ----------
        enabled : bool
            Boolean value to enable/disable writing of V2 page headers.

        Returns
        -------
        Self
        """
        self.builder.write_v2_headers(enabled)
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
        self.builder.dictionary_policy(val)
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
        self.builder.utc_timestamps(enabled)
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
        self.builder.write_arrow_schema(enabled)
        return self

    cpdef ParquetWriterOptions build(self):
        """
        Options member once it's built

        Returns
        -------
        ParquetWriterOptions
        """
        cdef ParquetWriterOptions parquet_options = ParquetWriterOptions.__new__(
            ParquetWriterOptions
        )
        parquet_options.options = move(self.builder.build())
        return parquet_options


cdef class BufferArrayFromVector:
    @staticmethod
    cdef BufferArrayFromVector from_unique_ptr(
        unique_ptr[vector[uint8_t]] in_vec
    ):
        cdef BufferArrayFromVector buf = BufferArrayFromVector()
        buf.in_vec = move(in_vec)
        buf.length = dereference(buf.in_vec).size()
        return buf

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        cdef Py_ssize_t itemsize = sizeof(uint8_t)

        self.shape[0] = self.length
        self.strides[0] = 1

        buffer.buf = dereference(self.in_vec).data()

        buffer.format = NULL  # byte
        buffer.internal = NULL
        buffer.itemsize = itemsize
        buffer.len = self.length * itemsize   # product(shape) * itemsize
        buffer.ndim = 1
        buffer.obj = self
        buffer.readonly = 0
        buffer.shape = self.shape
        buffer.strides = self.strides
        buffer.suboffsets = NULL

    def __releasebuffer__(self, Py_buffer *buffer):
        pass


cpdef BufferArrayFromVector write_parquet(ParquetWriterOptions options):
    """
    Writes a set of columns to parquet format.

    Parameters
    ----------
    options : ParquetWriterOptions
        Settings for controlling writing behavior

    Returns
    -------
    BufferArrayFromVector
        A blob that contains the file metadata
        (parquet FileMetadata thrift message) if requested in
        parquet_writer_options (empty blob otherwise).
    """
    cdef parquet_writer_options c_options = options.options
    cdef unique_ptr[vector[uint8_t]] c_result

    with nogil:
        c_result = cpp_write_parquet(c_options)

    return BufferArrayFromVector.from_unique_ptr(move(c_result))
