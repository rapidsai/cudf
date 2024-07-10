# Copyright (c) 2024, NVIDIA CORPORATION.

from libcpp cimport bool
from libcpp.map cimport map
from libcpp.string cimport string
from libcpp.utility cimport move
from libcpp.vector cimport vector

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
from cudf._lib.pylibcudf.libcudf.types cimport data_type, size_type
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
    str decimal = ".",
    str comment = None,
    bool delim_whitespace = False,
    bool skipinitialspace = False,
    bool skip_blank_lines = True,
    quote_style quoting = quote_style.MINIMAL,
    str quotechar = '"',
    bool doublequote = True,
    list parse_dates = None,
    list parse_hex = None,
    # Technically this should be dict/list
    # but using a fused type prevents using None as default
    object dtypes = None,
    list true_values = None,
    list false_values = None,
    list na_values = None,
    bool keep_default_na = True,
    bool na_filter = True,
    bool dayfirst = False,
    # Note: These options are supported by the libcudf reader
    # but are not exposed here (since there is no demand for them
    # on the Python side yet)
    # bool detect_whitespace_around_quotes = False,
    # DataType timestamp_type = DataType(type_id.EMPTY),
):
    """Reads an CSV file into a :py:class:`~.types.TableWithMetadata`.

    Parameters
    ----------
    source_info : SourceInfo
        The SourceInfo to read the CSV file from.
    compression : compression_type, default CompressionType.AUTO
        The compression format of the CSV source.
    byte_range_offset : size_type, default 0
        Number of bytes to skip from source start.
    byte_range_size : size_type, default 0
        Number of bytes to read. By default, will read all bytes.
    col_names : list, default None
        The column names to use.
    prefix : string, default ""
        The prefix to apply to the column names.
    mangle_dupe_cols : bool, default True
        If True, rename duplicate column names.
    usecols : list, default None
        Specify the string column names/integer column indices of columns to be read.
    nrows : size_type, default -1
        The number of rows to read.
    skiprows : size_type, default 0
        The number of rows to skip from the start before reading
    skipfooter : size_type, default 0
        The number of rows to skip from the end
    header : size_type, default 0
        The index of the row that will be used for header names.
        Pass -1 to use default column names.
    lineterminator : str, default "\n"
        The character used to determine the end of a line.
    delimiter : str, default ","
        The character used to separate fields in a row.
    thousands : str, default None
        The character used as the thousands separator.
        Cannot match delimiter.
    decimal : str, default "."
        The character used as the decimal separator.
        Cannot match delimiter.
    comment : str, default None
        The character used to identify the start of a comment line.
        (which will be skipped by the reader)
    delim_whitespace : bool, default False
        If True, treat whitespace as the field delimiter.
    skipinitialspace : bool, default False
        If True, skip whitespace after the delimiter.
    skip_blank_lines : bool, default True
        If True, ignore empty lines (otherwise line values are parsed as null).
    quoting : QuoteStyle, default QuoteStyle.MINIMAL
        The quoting style used in the input CSV data. One of
        { QuoteStyle.MINIMAL, QuoteStyle.ALL, QuoteStyle.NONNUMERIC, QuoteStyle.NONE }
    quotechar : str, default '"'
        The character used to indicate quoting.
    doublequote : bool, default True
        If True, a quote inside a value is double-quoted.
    parse_dates : list, default None
        A list of integer column indices/string column names
        of columns to read as datetime.
    parse_hex : list, default None
        A list of integer column indices/string column names
        of columns to read as hexadecimal.
    dtypes : Union[Dict[str, DataType], List[DataType]], default None
        A list of data types or a dictionary mapping column names
        to a DataType.
    true_values : List[str], default None
        A list of additional values to recognize as True.
    false_values : List[str], default None
        A list of additional values to recognize as False.
    na_values : List[str], default None
        A list of additional values to recognize as null.
    keep_default_na : bool, default True
        Whether to keep the built-in default N/A values.
    na_filter : bool, default True
        Whether to detect missing values. If False, can
        improve performance.
    dayfirst : bool, default False
        If True, interpret dates as being in the DD/MM format.

    Returns
    -------
    TableWithMetadata
        The Table and its corresponding metadata (column names) that were read in.
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
    if isinstance(dtypes, list):
        for dtype in dtypes:
            if not isinstance(dtype, DataType):
                raise TypeError("If passing list to read_csv, "
                                "all elements must be of type `DataType`!")
            c_dtypes_list.push_back((<DataType>dtype).c_obj)
        options.set_dtypes(c_dtypes_list)
    elif isinstance(dtypes, dict):
        # dtypes_t is dict
        for k, v in dtypes.items():
            k_str = str(k).encode()
            if not isinstance(v, DataType):
                raise TypeError("If passing dict to read_csv, "
                                "all values must be of type `DataType`!")
            c_dtypes_map[k_str] = (<DataType>v).c_obj
        options.set_dtypes(c_dtypes_map)
    elif dtypes is not None:
        raise TypeError("dtypes must either by a list/dict")

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
