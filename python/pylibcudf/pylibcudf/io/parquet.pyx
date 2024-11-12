# Copyright (c) 2024, NVIDIA CORPORATION.
from cython.operator cimport dereference
from libc.stdint cimport int64_t
from libcpp cimport bool
from libcpp.string cimport string
from libcpp.utility cimport move
from libcpp.vector cimport vector
from pylibcudf.expressions cimport Expression
from pylibcudf.io.types cimport SourceInfo, TableWithMetadata
from pylibcudf.libcudf.expressions cimport expression
from pylibcudf.libcudf.io.parquet cimport (
    chunked_parquet_reader as cpp_chunked_parquet_reader,
    parquet_reader_options,
    read_parquet as cpp_read_parquet,
)
from pylibcudf.libcudf.io.types cimport table_with_metadata
from pylibcudf.libcudf.types cimport size_type

__all__ = ["ChunkedParquetReader", "read_parquet"]


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
