# Copyright (c) 2024, NVIDIA CORPORATION.

from libcpp cimport bool
from libcpp.map cimport map
from libcpp.string cimport string
from libcpp.utility cimport move
from libcpp.vector cimport vector

from cudf._lib.pylibcudf.io.csv cimport dtypes_t
from cudf._lib.pylibcudf.io.types cimport SourceInfo, TableWithMetadata
from cudf._lib.pylibcudf.libcudf.io.csv cimport (
    csv_reader_options,
    read_csv as cpp_read_csv,
)
from cudf._lib.pylibcudf.libcudf.io.types cimport (
    compression_type,
    quote_style,
    table_with_metadata,
)
from cudf._lib.pylibcudf.libcudf.types cimport data_type, size_type, type_id
from cudf._lib.pylibcudf.types cimport DataType


cpdef TableWithMetadata read_csv(
    SourceInfo source_info,
    compression_type compression = compression_type.AUTO,
    size_t byte_range_offset = 0,
    size_t byte_range_size = 0,
    list col_names = None,
    str prefix = "",
    bool mangle_dupe_cols = True,
    list usecols = None,
    size_type nrows = -1,
    size_type skiprows = 0,
    size_type skipfooter = 0,
    size_type header = 0,
    str lineterminator = "\n",
    str delimiter = None,
    str thousands = None,
    str decimal = None,
    str comment = None,
    bool delim_whitespace = False,
    bool skipinitialspace = False,
    bool skip_blank_lines = True,
    quote_style quoting = quote_style.MINIMAL,
    str quotechar = '"',
    bool doublequote = True,
    bool detect_whitespace_around_quotes = False,
    list parse_dates = None,
    list parse_hex = None,
    dtypes_t dtypes = None,
    list true_values = None,
    list false_values = None,
    list na_values = None,
    bool keep_default_na = True,
    bool na_filter = True,
    bool dayfirst = False,
    DataType timestamp_type = DataType(type_id.EMPTY)
):
    """

    """
    cdef vector[string] c_names
    cdef vector[int] c_use_cols_indexes
    cdef vector[string] c_use_cols_names
    cdef vector[string] c_parse_dates_names
    cdef vector[int] c_parse_dates_indexes
    cdef vector[int] c_parse_hex_names
    cdef vector[int] c_parse_hex_indexes
    cdef vector[data_type] c_dtypes_list
    cdef map[string, data_type] c_dtypes_map
    cdef vector[string] c_true_values
    cdef vector[string] c_false_values
    cdef vector[string] c_na_values

    cdef csv_reader_options options = move(
        csv_reader_options.builder(source_info.c_obj)
        .compression(compression)
        .mangle_dupe_cols(mangle_dupe_cols)
        .byte_range_offset(byte_range_offset)
        .byte_range_size(byte_range_size)
        .nrows(nrows)
        .skiprows(skiprows)
        .skipfooter(skipfooter)
        .quoting(quoting)
        .lineterminator(ord(lineterminator))
        .quotechar(ord(quotechar))
        .decimal(ord(decimal))
        .delim_whitespace(delim_whitespace)
        .skipinitialspace(skipinitialspace)
        .skip_blank_lines(skip_blank_lines)
        .doublequote(doublequote)
        .keep_default_na(keep_default_na)
        .na_filter(na_filter)
        .dayfirst(dayfirst)
        .build()
    )

    options.set_header(header)

    if col_names is not None:
        c_names.reserve(len(col_names))
        for name in col_names:
            c_names.push_back(str(name).encode())
        options.set_names(c_names)

    if prefix is not None:
        options.set_prefix(prefix.encode())

    if usecols is not None:
        all_int = all([isinstance(col, int) for col in usecols])
        if all_int:
            c_use_cols_indexes.reserve(len(usecols))
            c_use_cols_indexes = usecols
            options.set_use_cols_indexes(c_use_cols_indexes)
        else:
            c_use_cols_names.reserve(len(usecols))
            for col_name in usecols:
                c_use_cols_names.push_back(
                    str(col_name).encode()
                )
            options.set_use_cols_names(c_use_cols_names)

    if delimiter is not None:
        options.set_delimiter(ord(delimiter))

    if thousands is not None:
        options.set_thousands(ord(thousands))

    if comment is not None:
        options.set_comment(ord(comment))

    if parse_dates is not None:
        for col in parse_dates:
            if isinstance(col, str):
                c_parse_dates_names.push_back(col.encode())
            elif isinstance(col, int):
                c_parse_dates_indexes.push_back(col)
            else:
                raise NotImplementedError(
                    "`parse_dates`: Must pass a list of column names/indices")

        # Set both since users are allowed to mix column names and indices
        options.set_parse_dates(c_parse_dates_names)
        options.set_parse_dates(c_parse_dates_indexes)

    if parse_hex is not None:
        for col in parse_hex:
            if isinstance(col, str):
                c_parse_hex_names.push_back(col.encode())
            elif isinstance(col, int):
                c_parse_hex_indexes.push_back(col)
            else:
                raise NotImplementedError(
                    "`parse_hex`: Must pass a list of column names/indices")
        # Set both since users are allowed to mix column names and indices
        options.set_parse_hex(c_parse_hex_names)
        options.set_parse_hex(c_parse_hex_indexes)

    cdef string k_str
    if dtypes is not None:
        if dtypes_t is list:
            for dtype in dtypes:
                if not isinstance(dtype, DataType):
                    raise TypeError("If passing list to read_csv, "
                                    "all elements must be of type `DataType`!")
                c_dtypes_list.push_back((<DataType>dtype).c_obj)
            options.set_dtypes(c_dtypes_list)
        else:
            # dtypes_t is dict
            for k, v in dtypes.items():
                k_str = str(k).encode()
                if not isinstance(v, DataType):
                    raise TypeError("If passing dict to read_csv, "
                                    "all values must be of type `DataType`!")
                c_dtypes_map[k_str] = (<DataType>v).c_obj
            options.set_dtypes(c_dtypes_map)

    if true_values is not None:
        c_true_values.reserve(len(true_values))
        for tv in true_values:
            if not isinstance(tv, str):
                raise TypeError("true_values must be a list of str!")
            c_true_values.push_back(tv.encode())
        options.set_true_values(c_true_values)

    if false_values is not None:
        c_false_values.reserve(len(false_values))
        for fv in false_values:
            if not isinstance(fv, str):
                raise TypeError("false_values must be a list of str!")
            c_false_values.push_back(fv.encode())
        options.set_false_values(c_false_values)

    if na_values is not None:
        c_na_values.reserve(len(na_values))
        for nv in na_values:
            if not isinstance(nv, str):
                raise TypeError("na_values must be a list of str!")
            c_na_values.push_back(nv.encode())
        options.set_na_values(c_na_values)

    cdef table_with_metadata c_result
    with nogil:
        c_result = move(cpp_read_csv(options))

    return TableWithMetadata.from_libcudf(c_result)
