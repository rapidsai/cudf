# Copyright (c) 2024, NVIDIA CORPORATION.
from libcpp cimport bool
from libcpp.limits cimport numeric_limits
from libcpp.map cimport map
from libcpp.string cimport string
from libcpp.utility cimport move
from libcpp.vector cimport vector
from pylibcudf.concatenate cimport concatenate
from pylibcudf.io.types cimport SinkInfo, SourceInfo, TableWithMetadata
from pylibcudf.libcudf.io.json cimport (
    json_reader_options,
    json_recovery_mode_t,
    json_writer_options,
    read_json as cpp_read_json,
    schema_element,
    write_json as cpp_write_json,
)
from pylibcudf.libcudf.io.types cimport (
    compression_type,
    table_metadata,
    table_with_metadata,
)
from pylibcudf.libcudf.types cimport data_type, size_type
from pylibcudf.types cimport DataType

__all__ = ["chunked_read_json", "read_json", "write_json"]

cdef map[string, schema_element] _generate_schema_map(list dtypes):
    cdef map[string, schema_element] schema_map
    cdef schema_element s_elem
    cdef string c_name

    for name, dtype, child_dtypes in dtypes:
        if not (isinstance(name, str) and
                isinstance(dtype, DataType) and
                isinstance(child_dtypes, list)):

            raise ValueError("Must pass a list of a tuple containing "
                             "(column_name, column_dtype, list of child_dtypes)")

        c_name = <str>name.encode()

        s_elem.type = (<DataType>dtype).c_obj
        s_elem.child_types = _generate_schema_map(child_dtypes)

        schema_map[c_name] = s_elem
    return schema_map


cdef json_reader_options _setup_json_reader_options(
        SourceInfo source_info,
        list dtypes,
        compression_type compression,
        bool lines,
        size_t byte_range_offset,
        size_t byte_range_size,
        bool keep_quotes,
        bool mixed_types_as_string,
        bool prune_columns,
        json_recovery_mode_t recovery_mode):

    cdef vector[data_type] types_vec
    cdef json_reader_options opts = (
        json_reader_options.builder(source_info.c_obj)
        .compression(compression)
        .lines(lines)
        .byte_range_offset(byte_range_offset)
        .byte_range_size(byte_range_size)
        .recovery_mode(recovery_mode)
        .build()
    )

    if dtypes is not None:
        if isinstance(dtypes[0], tuple):
            opts.set_dtypes(move(_generate_schema_map(dtypes)))
        else:
            for dtype in dtypes:
                types_vec.push_back((<DataType>dtype).c_obj)
            opts.set_dtypes(types_vec)

    opts.enable_keep_quotes(keep_quotes)
    opts.enable_mixed_types_as_string(mixed_types_as_string)
    opts.enable_prune_columns(prune_columns)
    return opts


cpdef tuple chunked_read_json(
    SourceInfo source_info,
    list dtypes = None,
    compression_type compression = compression_type.AUTO,
    bool keep_quotes = False,
    bool mixed_types_as_string = False,
    bool prune_columns = False,
    json_recovery_mode_t recovery_mode = json_recovery_mode_t.FAIL,
    int chunk_size=100_000_000,
):
    """Reads an JSON file into a :py:class:`~.types.TableWithMetadata`.

    Parameters
    ----------
    source_info : SourceInfo
        The SourceInfo object to read the JSON file from.
    dtypes : list, default None
        Set data types for the columns in the JSON file.

        Each element of the list has the format
        (column_name, column_dtype, list of child dtypes), where
        the list of child dtypes is an empty list if the child is not
        a nested type (list or struct dtype), and is of format
        (column_child_name, column_child_type, list of grandchild dtypes).
    compression: CompressionType, default CompressionType.AUTO
        The compression format of the JSON source.
    keep_quotes : bool, default False
        Whether the reader should keep quotes of string values.
    mixed_types_as_string : bool, default False
        If True, mixed type columns are returned as string columns.
        If `False` parsing mixed type columns will thrown an error.
    prune_columns : bool, default False
        Whether to only read columns specified in dtypes.
    recover_mode : JSONRecoveryMode, default JSONRecoveryMode.FAIL
        Whether to raise an error or set corresponding values to null
        when encountering an invalid JSON line.
    chunk_size : int, default 100_000_000 bytes.
        The number of bytes to be read in chunks.
        The chunk_size should be set to at least row_size.

    Returns
    -------
    tuple
        A tuple of (columns, column_name, child_names)
    """
    cdef size_type c_range_size = (
        chunk_size if chunk_size is not None else 0
    )
    cdef json_reader_options opts = _setup_json_reader_options(
        source_info=source_info,
        dtypes=dtypes,
        compression=compression,
        lines=True,
        byte_range_offset=0,
        byte_range_size=0,
        keep_quotes=keep_quotes,
        mixed_types_as_string=mixed_types_as_string,
        prune_columns=prune_columns,
        recovery_mode=recovery_mode,
    )

    # Read JSON
    cdef table_with_metadata c_result

    final_columns = []
    meta_names = None
    child_names = None
    i = 0
    while True:
        opts.set_byte_range_offset(c_range_size * i)
        opts.set_byte_range_size(c_range_size)

        try:
            with nogil:
                c_result = move(cpp_read_json(opts))
        except (ValueError, OverflowError):
            break
        if meta_names is None:
            meta_names = [info.name.decode() for info in c_result.metadata.schema_info]
        if child_names is None:
            child_names = TableWithMetadata._parse_col_names(
                c_result.metadata.schema_info
            )
        new_chunk = [
            col for col in TableWithMetadata.from_libcudf(
                c_result).columns
        ]

        if len(final_columns) == 0:
            final_columns = new_chunk
        else:
            for col_idx in range(len(meta_names)):
                final_columns[col_idx] = concatenate(
                    [final_columns[col_idx], new_chunk[col_idx]]
                )
                # Must drop any residual GPU columns to save memory
                new_chunk[col_idx] = None
        i += 1
    return (final_columns, meta_names, child_names)


cpdef TableWithMetadata read_json(
    SourceInfo source_info,
    list dtypes = None,
    compression_type compression = compression_type.AUTO,
    bool lines = False,
    size_t byte_range_offset = 0,
    size_t byte_range_size = 0,
    bool keep_quotes = False,
    bool mixed_types_as_string = False,
    bool prune_columns = False,
    json_recovery_mode_t recovery_mode = json_recovery_mode_t.FAIL,
):
    """Reads an JSON file into a :py:class:`~.types.TableWithMetadata`.

    Parameters
    ----------
    source_info : SourceInfo
        The SourceInfo object to read the JSON file from.
    dtypes : list, default None
        Set data types for the columns in the JSON file.

        Each element of the list has the format
        (column_name, column_dtype, list of child dtypes), where
        the list of child dtypes is an empty list if the child is not
        a nested type (list or struct dtype), and is of format
        (column_child_name, column_child_type, list of grandchild dtypes).
    compression: CompressionType, default CompressionType.AUTO
        The compression format of the JSON source.
    byte_range_offset : size_t, default 0
        Number of bytes to skip from source start.
    byte_range_size : size_t, default 0
        Number of bytes to read. By default, will read all bytes.
    keep_quotes : bool, default False
        Whether the reader should keep quotes of string values.
    mixed_types_as_string : bool, default False
        If True, mixed type columns are returned as string columns.
        If `False` parsing mixed type columns will thrown an error.
    prune_columns : bool, default False
        Whether to only read columns specified in dtypes.
    recover_mode : JSONRecoveryMode, default JSONRecoveryMode.FAIL
        Whether to raise an error or set corresponding values to null
        when encountering an invalid JSON line.

    Returns
    -------
    TableWithMetadata
        The Table and its corresponding metadata (column names) that were read in.
    """
    cdef json_reader_options opts = _setup_json_reader_options(
        source_info=source_info,
        dtypes=dtypes,
        compression=compression,
        lines=lines,
        byte_range_offset=byte_range_offset,
        byte_range_size=byte_range_size,
        keep_quotes=keep_quotes,
        mixed_types_as_string=mixed_types_as_string,
        prune_columns=prune_columns,
        recovery_mode=recovery_mode,
    )

    # Read JSON
    cdef table_with_metadata c_result

    with nogil:
        c_result = move(cpp_read_json(opts))

    return TableWithMetadata.from_libcudf(c_result)


cpdef void write_json(
    SinkInfo sink_info,
    TableWithMetadata table_w_meta,
    str na_rep = "",
    bool include_nulls = False,
    bool lines = False,
    size_type rows_per_chunk = numeric_limits[size_type].max(),
    str true_value = "true",
    str false_value = "false"
):
    """
    Writes a :py:class:`~pylibcudf.table.Table` to JSON format.

    Parameters
    ----------
    sink_info: SinkInfo
        The SinkInfo object to write the JSON to.
    table_w_meta: TableWithMetadata
        The TableWithMetadata object containing the Table to write
    na_rep: str, default ""
        The string representation for null values.
    include_nulls: bool, default False
        Enables/Disables output of nulls as 'null'.
    lines: bool, default False
        If `True`, write output in the JSON lines format.
    rows_per_chunk: size_type, defaults to length of the input table
        The maximum number of rows to write at a time.
    true_value: str, default "true"
        The string representation for values != 0 in INT8 types.
    false_value: str, default "false"
        The string representation for values == 0 in INT8 types.
    """
    cdef table_metadata tbl_meta = table_w_meta.metadata
    cdef string na_rep_c = na_rep.encode()

    cdef json_writer_options options = (
        json_writer_options.builder(sink_info.c_obj, table_w_meta.tbl.view())
        .metadata(tbl_meta)
        .na_rep(na_rep_c)
        .include_nulls(include_nulls)
        .lines(lines)
        .build()
    )

    if rows_per_chunk != numeric_limits[size_type].max():
        options.set_rows_per_chunk(rows_per_chunk)
    if true_value != "true":
        options.set_true_value(<string>true_value.encode())
    if false_value != "false":
        options.set_false_value(<string>false_value.encode())

    with nogil:
        cpp_write_json(options)
