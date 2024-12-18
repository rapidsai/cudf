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

__all__ = [
    "chunked_read_json",
    "read_json",
    "write_json",
    "JsonReaderOptions",
    "JsonReaderOptionsBuilder"
]

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


cpdef JsonReaderOptions _setup_json_reader_options(
        SourceInfo source_info,
        list dtypes,
        compression_type compression = compression_type.AUTO,
        bool lines = False,
        size_t byte_range_offset = 0,
        size_t byte_range_size = 0,
        bool keep_quotes = False,
        bool mixed_types_as_string = False,
        bool prune_columns = False,
        json_recovery_mode_t recovery_mode = json_recovery_mode_t.FAIL,
        dict extra_parameters=None,
):
    options = (
        JsonReaderOptions.builder(source_info)
        .compression(compression)
        .lines(lines)
        .byte_range_offset(byte_range_offset)
        .byte_range_size(byte_range_size)
        .recovery_mode(recovery_mode)
        .build()
    )

    if dtypes is not None:
        options.set_dtypes(dtypes)

    options.enable_keep_quotes(keep_quotes)
    options.enable_mixed_types_as_string(mixed_types_as_string)
    options.enable_prune_columns(prune_columns)

    # These hidden options are subjected to change without deprecation cycle.
    # These are used to test libcudf JSON reader features, not used in cuDF.
    if extra_parameters is not None:
        for key, value in extra_parameters.items():
            if key == 'delimiter':
                options.set_delimiter(value)
            elif key == 'dayfirst':
                options.enable_dayfirst(value)
            elif key == 'experimental':
                options.enable_experimental(value)
            elif key == 'normalize_single_quotes':
                options.enable_normalize_single_quotes(value)
            elif key == 'normalize_whitespace':
                options.enable_normalize_whitespace(value)
            elif key == 'strict_validation':
                options.set_strict_validation(value)
            elif key == 'allow_unquoted_control_chars':
                options.allow_unquoted_control_chars(value)
            elif key == 'allow_numeric_leading_zeros':
                options.allow_numeric_leading_zeros(value)
            elif key == 'allow_nonnumeric_numbers':
                options.allow_nonnumeric_numbers(value)
            elif key == 'na_values':
                options.set_na_values(value)
            else:
                raise ValueError(
                    "cudf engine doesn't support the "
                    f"'{key}' keyword argument for read_json"
                )
    return options


cdef class JsonReaderOptions:
    """
    The settings to use for ``read_json``

    For details, see `:cpp:class:`cudf::io::json_reader_options`
    """
    @staticmethod
    def builder(SourceInfo source):
        """
        Create a JsonReaderOptionsBuilder object

        For details, see :cpp:func:`cudf::io::json_reader_options::builder`

        Parameters
        ----------
        sink : SourceInfo
            The source to read the JSON file from.

        Returns
        -------
        JsonReaderOptionsBuilder
            Builder to build JsonReaderOptions
        """
        cdef JsonReaderOptionsBuilder json_builder = (
            JsonReaderOptionsBuilder.__new__(JsonReaderOptionsBuilder)
        )
        json_builder.c_obj = json_reader_options.builder(source.c_obj)
        json_builder.source = source
        return json_builder

    cpdef void set_dtypes(self, list types):
        """
        Set data types for columns to be read.

        Parameters
        ----------
        types : list
            List of dtypes or a list of tuples of
            column names, dtypes, and list of tuples
            (to support nested column hierarchy)

        Returns
        -------
        None
        """
        cdef vector[data_type] types_vec
        if isinstance(types[0], tuple):
            self.c_obj.set_dtypes(_generate_schema_map(types))
        else:
            types_vec.reserve(len(types))
            for dtype in types:
                types_vec.push_back((<DataType>dtype).c_obj)
            self.c_obj.set_dtypes(types_vec)

    cpdef void enable_keep_quotes(self, bool keep_quotes):
        """
        Set whether the reader should keep quotes of string values.

        Parameters
        ----------
        keep_quotes : bool
           Boolean value to indicate whether the reader should
           keep quotes of string values

        Returns
        -------
        None
        """
        self.c_obj.enable_keep_quotes(keep_quotes)

    cpdef void enable_mixed_types_as_string(self, bool mixed_types_as_string):
        """
        Set whether to parse mixed types as a string column.
        Also enables forcing to read a struct as string column using schema.

        Parameters
        ----------
        mixed_types_as_string : bool
           Boolean value to enable/disable parsing mixed types
           as a string column

        Returns
        -------
        None
        """
        self.c_obj.enable_mixed_types_as_string(mixed_types_as_string)

    cpdef void enable_prune_columns(self, bool prune_columns):
        """
        Set whether to prune columns on read, selected
        based on the ``set_dtypes`` option.

        Parameters
        ----------
        prune_columns : bool
           When set as true, if the reader options include
           ``set_dtypes``, then the reader will only return those
           columns which are mentioned in ``set_dtypes``. If false,
           then all columns are returned, independent of the
           ``set_dtypes`` setting.

        Returns
        -------
        None
        """
        self.c_obj.enable_prune_columns(prune_columns)

    cpdef void set_byte_range_offset(self, size_t offset):
        """
        Set number of bytes to skip from source start.

        Parameters
        ----------
        offset : size_t
            Number of bytes of offset

        Returns
        -------
        None
        """
        self.c_obj.set_byte_range_offset(offset)

    cpdef void set_byte_range_size(self, size_t size):
        """
        Set number of bytes to read.

        Parameters
        ----------
        size : size_t
            Number of bytes to read

        Returns
        -------
        None
        """
        self.c_obj.set_byte_range_size(size)

    cpdef void enable_lines(self, bool val):
        """
        Set whether to read the file as a json object per line.

        Parameters
        ----------
        val : bool
            Boolean value to enable/disable the option
            to read each line as a json object

        Returns
        -------
        None
        """
        self.c_obj.enable_lines(val)

    # These hidden options are subjected to change without deprecation cycle.
    # These are used to test libcudf JSON reader features, not used in cuDF.

    cpdef void set_delimiter(self, str val):
        self.c_obj.set_delimiter(val.encode())

    cpdef void enable_dayfirst(self, bool val):
        self.c_obj.enable_dayfirst(val)

    cpdef void enable_experimental(self, bool val):
        self.c_obj.enable_experimental(val)

    cpdef void enable_normalize_single_quotes(self, bool val):
        self.c_obj.enable_normalize_single_quotes(val)

    cpdef void enable_normalize_whitespace(self, bool val):
        self.c_obj.enable_normalize_whitespace(val)

    cpdef void set_strict_validation(self, bool val):
        self.c_obj.set_strict_validation(val)

    cpdef void allow_unquoted_control_chars(self, bool val):
        self.c_obj.allow_unquoted_control_chars(val)

    cpdef void allow_numeric_leading_zeros(self, bool val):
        self.c_obj.allow_numeric_leading_zeros(val)

    cpdef void allow_nonnumeric_numbers(self, bool val):
        self.c_obj.allow_nonnumeric_numbers(val)

    cpdef void set_na_values(self, list vals):
        cdef vector[string] vec
        for val in vals:
            if isinstance(val, str):
                vec.push_back(val.encode())
        self.c_obj.set_na_values(vec)


cdef class JsonReaderOptionsBuilder:
    cpdef JsonReaderOptionsBuilder compression(self, compression_type compression):
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

    cpdef JsonReaderOptionsBuilder lines(self, bool val):
        """
        Set whether to read the file as a json object per line.

        Parameters
        ----------
        val : bool
            Boolean value to enable/disable the option
            to read each line as a json object

        Returns
        -------
        Self
        """
        self.c_obj.lines(val)
        return self

    cpdef JsonReaderOptionsBuilder keep_quotes(self, bool val):
        """
        Set whether the reader should keep quotes of string values.

        Parameters
        ----------
        val : bool
            Boolean value to indicate whether the
            reader should keep quotes of string values

        Returns
        -------
        Self
        """
        self.c_obj.keep_quotes(val)
        return self

    cpdef JsonReaderOptionsBuilder byte_range_offset(self, size_t byte_range_offset):
        """
        Set number of bytes to skip from source start.

        Parameters
        ----------
        byte_range_offset : size_t
            Number of bytes of offset

        Returns
        -------
        Self
        """
        self.c_obj.byte_range_offset(byte_range_offset)
        return self

    cpdef JsonReaderOptionsBuilder byte_range_size(self, size_t byte_range_size):
        """
        Set number of bytes to read.

        Parameters
        ----------
        byte_range_size : size_t
            Number of bytes to read

        Returns
        -------
        Self
        """
        self.c_obj.byte_range_size(byte_range_size)
        return self

    cpdef JsonReaderOptionsBuilder recovery_mode(
        self,
        json_recovery_mode_t recovery_mode
    ):
        """
        Specifies the JSON reader's behavior on invalid JSON lines.

        Parameters
        ----------
        recovery_mode : json_recovery_mode_t
            An enum value to indicate the JSON reader's
            behavior on invalid JSON lines.

        Returns
        -------
        Self
        """
        self.c_obj.recovery_mode(recovery_mode)
        return self

    cpdef build(self):
        """Create a JsonReaderOptions object"""
        cdef JsonReaderOptions json_options = JsonReaderOptions.__new__(
            JsonReaderOptions
        )
        json_options.c_obj = move(self.c_obj.build())
        json_options.source = self.source
        return json_options


cpdef tuple chunked_read_json(
    JsonReaderOptions options,
    int chunk_size=100_000_000,
):
    """
    Reads chunks of a JSON file into a :py:class:`~.types.TableWithMetadata`.

    Parameters
    ----------
    options : JsonReaderOptions
        Settings for controlling reading behavior
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
    cdef table_with_metadata c_result

    final_columns = []
    meta_names = None
    child_names = None
    i = 0
    while True:
        options.enable_lines(True)
        options.set_byte_range_offset(c_range_size * i)
        options.set_byte_range_size(c_range_size)

        try:
            with nogil:
                c_result = move(cpp_read_json(options.c_obj))
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
    JsonReaderOptions options
):
    """
    Read from JSON format.

    The source to read from and options are encapsulated
    by the `options` object.

    For details, see :cpp:func:`read_json`.

    Parameters
    ----------
    options: JsonReaderOptions
        Settings for controlling reading behavior

    Returns
    -------
    TableWithMetadata
        The Table and its corresponding metadata (column names) that were read in.
    """
    cdef table_with_metadata c_result

    with nogil:
        c_result = move(cpp_read_json(options.c_obj))

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
