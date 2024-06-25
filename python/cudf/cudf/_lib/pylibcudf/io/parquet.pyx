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
from cudf._lib.pylibcudf.libcudf.types cimport size_type, type_id
from cudf._lib.pylibcudf.types cimport DataType


cdef parquet_reader_options _setup_parquet_reader_options(
    SourceInfo source_info,
    list columns = None,
    list row_groups = None,
    Expression filters = None,
    bool convert_strings_to_categories = False,
    bool use_pandas_metadata = True,
    # ReaderColumnSchema reader_column_schema = None,
    int64_t skip_rows = 0,
    size_type num_rows = -1,
    DataType timestamp_type = DataType(type_id.EMPTY)
):
    cdef vector[string] col_vec
    cdef parquet_reader_options opts = move(
        parquet_reader_options.builder(source_info.c_obj)
        .convert_strings_to_categories(convert_strings_to_categories)
        .use_pandas_metadata(use_pandas_metadata)
        .use_arrow_schema(True)
        .timestamp_type(timestamp_type.c_obj)
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

    def __init__(
        self,
        SourceInfo source_info,
        list columns=None,
        list row_groups=None,
        bool use_pandas_metadata=True,
        bool convert_strings_to_categories=False,
        int64_t skip_rows = 0,
        size_type num_rows = -1,
        DataType timestamp_type = DataType(type_id.EMPTY),
        size_t chunk_read_limit=0,
        size_t pass_read_limit=1024000000
    ):

        cdef parquet_reader_options opts = move(
            _setup_parquet_reader_options(
                source_info,
                columns,
                row_groups,
                filters=None,
                convert_strings_to_categories=convert_strings_to_categories,
                use_pandas_metadata=use_pandas_metadata,
                skip_rows=skip_rows,
                num_rows=num_rows,
                timestamp_type=timestamp_type
            )
        )

        with nogil:
            self.reader.reset(
                new cpp_chunked_parquet_reader(
                    chunk_read_limit,
                    pass_read_limit,
                    opts
                )
            )

    def _has_next(self):
        cdef bool res
        with nogil:
            res = self.reader.get()[0].has_next()
        return res

    def _read_chunk(self):
        # Read Parquet
        cdef table_with_metadata c_result

        with nogil:
            c_result = move(self.reader.get()[0].read_chunk())

        return TableWithMetadata.from_libcudf(c_result)

cdef class ReaderColumnSchema:
    pass

cpdef read_parquet(
    SourceInfo source_info,
    list columns = None,
    list row_groups = None,
    Expression filters = None,
    bool convert_strings_to_categories = False,
    bool use_pandas_metadata = True,
    # ReaderColumnSchema reader_column_schema = None,
    int64_t skip_rows = 0,
    size_type num_rows = -1,
    DataType timestamp_type = DataType(type_id.EMPTY)
):
    """
    """
    cdef table_with_metadata c_result
    cdef parquet_reader_options opts = move(
        _setup_parquet_reader_options(
            source_info,
            columns,
            row_groups,
            filters,
            convert_strings_to_categories,
            use_pandas_metadata,
            skip_rows,
            num_rows,
            timestamp_type
        )
    )

    with nogil:
        c_result = move(cpp_read_parquet(opts))

    return TableWithMetadata.from_libcudf(c_result)
