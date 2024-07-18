# Copyright (c) 2024, NVIDIA CORPORATION.
from cython.operator cimport dereference
from libc.stdint cimport int64_t
from libcpp cimport bool
from libcpp.string cimport string
from libcpp.utility cimport move
from libcpp.vector cimport vector

from cudf._lib.pylibcudf.expressions cimport Expression
from cudf._lib.pylibcudf.io.types cimport SourceInfo, TableWithMetadata
from cudf._lib.pylibcudf.libcudf.expressions cimport expression
from cudf._lib.pylibcudf.libcudf.io.parquet cimport (
    chunked_parquet_reader as cpp_chunked_parquet_reader,
    parquet_reader_options,
    read_parquet as cpp_read_parquet,
)
from cudf._lib.pylibcudf.libcudf.io.types cimport table_with_metadata
from cudf._lib.pylibcudf.libcudf.types cimport size_type


cdef parquet_reader_options _setup_parquet_reader_options(
    SourceInfo source_info,
    list columns = None,
    list row_groups = None,
    Expression filters = None,
    bool convert_strings_to_categories = False,
    bool use_pandas_metadata = True,
    int64_t skip_rows = 0,
    size_type num_rows = -1,
    # ReaderColumnSchema reader_column_schema = None,
    # DataType timestamp_type = DataType(type_id.EMPTY)
):
    cdef vector[string] col_vec
    cdef parquet_reader_options opts = (
        parquet_reader_options.builder(source_info.c_obj)
        .convert_strings_to_categories(convert_strings_to_categories)
        .use_pandas_metadata(use_pandas_metadata)
        .use_arrow_schema(True)
        .build()
    )
    if row_groups is not None:
        opts.set_row_groups(row_groups)
    if num_rows != -1:
        opts.set_num_rows(num_rows)
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

    Parameters
    ----------
    source_info : SourceInfo
        The SourceInfo object to read the Parquet file from.
    columns : list, default None
        The names of the columns to be read
    row_groups : list[list[size_type]], default None
        List of row groups to be read.
    filters : Expression, default None
        An AST :py:class:`cudf._lib.pylibcudf.expression.Expression`
        to use for predicate pushdown.
    convert_strings_to_categories : bool, default False
        Whether to convert string columns to the category type
    use_pandas_metadata : bool, default True
        If True, return metadata about the index column in
        the per-file user metadata of the ``TableWithMetadata``
    skip_rows : int64_t, default 0
        The number of rows to skip from the start of the file.
    num_rows : size_type, default -1
        The number of rows to read. By default, read the entire file.
    """
    def __init__(
        self,
        SourceInfo source_info,
        list columns=None,
        list row_groups=None,
        bool use_pandas_metadata=True,
        bool convert_strings_to_categories=False,
        int64_t skip_rows = 0,
        size_type num_rows = -1,
        size_t chunk_read_limit=0,
        size_t pass_read_limit=1024000000
    ):

        cdef parquet_reader_options opts = _setup_parquet_reader_options(
            source_info,
            columns,
            row_groups,
            filters=None,
            convert_strings_to_categories=convert_strings_to_categories,
            use_pandas_metadata=use_pandas_metadata,
            skip_rows=skip_rows,
            num_rows=num_rows,
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
    size_type num_rows = -1,
    # Disabled, these aren't used by cudf-python
    # we should only add them back in if there's user demand
    # ReaderColumnSchema reader_column_schema = None,
    # DataType timestamp_type = DataType(type_id.EMPTY)
):
    """Reads an Parquet file into a :py:class:`~.types.TableWithMetadata`.

    Parameters
    ----------
    source_info : SourceInfo
        The SourceInfo object to read the Parquet file from.
    columns : list, default None
        The string names of the columns to be read.
    row_groups : list[list[size_type]], default None
        List of row groups to be read.
    filters : Expression, default None
        An AST :py:class:`cudf._lib.pylibcudf.expression.Expression`
        to use for predicate pushdown.
    convert_strings_to_categories : bool, default False
        Whether to convert string columns to the category type
    use_pandas_metadata : bool, default True
        If True, return metadata about the index column in
        the per-file user metadata of the ``TableWithMetadata``
    skip_rows : int64_t, default 0
        The number of rows to skip from the start of the file.
    num_rows : size_type, default -1
        The number of rows to read. By default, read the entire file.

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
        num_rows,
    )

    with nogil:
        c_result = move(cpp_read_parquet(opts))

    return TableWithMetadata.from_libcudf(c_result)
