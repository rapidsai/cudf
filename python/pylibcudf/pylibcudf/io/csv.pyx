# Copyright (c) 2024, NVIDIA CORPORATION.

from libcpp cimport bool
from libcpp.map cimport map

from libcpp.string cimport string
from libcpp.utility cimport move
from libcpp.vector cimport vector
from pylibcudf.io.types cimport SourceInfo, SinkInfo, TableWithMetadata
from pylibcudf.libcudf.io.csv cimport (
    csv_reader_options,
    csv_writer_options,
    read_csv as cpp_read_csv,
    write_csv as cpp_write_csv,
)

from pylibcudf.libcudf.io.types cimport (
    compression_type,
    quote_style,
    table_with_metadata,
)
from pylibcudf.libcudf.types cimport data_type, size_type
from pylibcudf.types cimport DataType
from pylibcudf.table cimport Table

__all__ = [
    "read_csv",
    "write_csv",
    "CsvWriterOptions",
    "CsvWriterOptionsBuilder",
    "CsvReaderOptions",
    "CsvReaderOptionsBuilder",
]

cdef tuple _process_parse_dates_hex(list cols):
    cdef vector[string] str_cols
    cdef vector[int] int_cols
    for col in cols:
        if isinstance(col, str):
            str_cols.push_back(col.encode())
        else:
            int_cols.push_back(col)
    return str_cols, int_cols

cdef vector[string] _make_str_vector(list vals):
    cdef vector[string] res
    for val in vals:
        res.push_back((<str?>val).encode())
    return res


cdef class CsvReaderOptions:
    @staticmethod
    def builder(SourceInfo source):
        cdef CsvReaderOptionsBuilder csv_builder = CsvReaderOptionsBuilder.__new__(
            CsvReaderOptionsBuilder
        )
        csv_builder.c_obj = csv_reader_options.builder(source.c_obj)
        csv_builder.source = source
        return csv_builder

    cpdef void set_header(self, size_type header):
        self.c_obj.set_header(header)

    cpdef void set_names(self, list col_names):
        cdef vector[string] vec
        for name in col_names:
            vec.push_back(name.encode())
        self.c_obj.set_names(vec)

    cpdef void set_prefix(self, str prefix):
        self.c_obj.set_prefix(prefix.encode())

    cpdef void set_use_cols_indexes(self, list col_indices):
        cdef vector[int] vec
        for i in col_indices:
            vec.push_back(i)
        self.c_obj.set_use_cols_indexes(vec)

    cpdef void set_use_cols_names(self, list col_names):
        cdef vector[string] vec
        for name in col_names:
            vec.push_back(name.encode())
        self.c_obj.set_use_cols_names(vec)

    cpdef void set_delimiter(self, str delimiter):
        self.c_obj.set_delimiter(ord(delimiter))

    cpdef void set_thousands(self, str thousands):
        self.c_obj.set_thousands(ord(thousands))

    cpdef void set_comment(self, str comment):
        self.c_obj.set_comment(ord(comment))

    cpdef void set_parse_dates(self, list val):
        cdef vector[string] vec_str
        cdef vector[int] vec_int
        if all([isinstance(date, str) for date in val]):
            for date in val:
                vec_str.push_back(date.encode())
            self.c_obj.set_parse_dates(vec_str)
        elif all([isinstance(date, int) for date in val]):
            for date in val:
                vec_int.push_back(date)
            self.c_obj.set_parse_dates(vec_int)
        else:
            raise TypeError("Must pass an int or str")

    cpdef void set_parse_hex(self, list val):
        cdef vector[string] vec_str
        cdef vector[int] vec_int
        if all([isinstance(hx, str) for hx in val]):
            for hx in val:
                vec_str.push_back(hx.encode())
            self.c_obj.set_parse_hex(vec_str)
        elif all([isinstance(hx, int) for hx in val]):
            for hx in val:
                vec_int.push_back(hx)
            self.c_obj.set_parse_hex(vec_int)
        else:
            raise TypeError("Must pass an int or str")

    cpdef void set_dtypes(self, object types):
        cdef map[string, data_type] dtype_map
        cdef vector[data_type] dtype_list
        if isinstance(types, dict):
            for name, dtype in types.items():
                dtype_map[str(name).encode()] = (<DataType>dtype).c_obj
            self.c_obj.set_dtypes(dtype_map)
        elif isinstance(types, list):
            for dtype in types:
                dtype_list.push_back((<DataType>dtype).c_obj)
            self.c_obj.set_dtypes(dtype_list)
        else:
            raise TypeError("Must pass an dict or list")

    cpdef void set_true_values(self, list true_values):
        cdef vector[string] vec
        for val in true_values:
            vec.push_back(val.encode())
        self.c_obj.set_true_values(vec)

    cpdef void set_false_values(self, list false_values):
        cdef vector[string] vec
        for val in false_values:
            vec.push_back(val.encode())
        self.c_obj.set_false_values(vec)

    cpdef void set_na_values(self, list na_values):
        cdef vector[string] vec
        for val in na_values:
            vec.push_back(val.encode())
        self.c_obj.set_na_values(vec)


cdef class CsvReaderOptionsBuilder:
    cpdef CsvReaderOptionsBuilder compression(self, compression_type compression):
        self.c_obj.compression(compression)
        return self

    cpdef CsvReaderOptionsBuilder mangle_dupe_cols(self, bool mangle_dupe_cols):
        self.c_obj.mangle_dupe_cols(mangle_dupe_cols)
        return self

    cpdef CsvReaderOptionsBuilder byte_range_offset(self, size_t byte_range_offset):
        self.c_obj.byte_range_offset(byte_range_offset)
        return self

    cpdef CsvReaderOptionsBuilder byte_range_size(self, size_t byte_range_size):
        self.c_obj.byte_range_size(byte_range_size)
        return self

    cpdef CsvReaderOptionsBuilder nrows(self, size_type nrows):
        self.c_obj.nrows(nrows)
        return self

    cpdef CsvReaderOptionsBuilder skiprows(self, size_type skiprows):
        self.c_obj.skiprows(skiprows)
        return self

    cpdef CsvReaderOptionsBuilder skipfooter(self, size_type skipfooter):
        self.c_obj.skipfooter(skipfooter)
        return self

    cpdef CsvReaderOptionsBuilder quoting(self, quote_style quoting):
        self.c_obj.quoting(quoting)
        return self

    cpdef CsvReaderOptionsBuilder lineterminator(self, str lineterminator):
        self.c_obj.lineterminator(ord(lineterminator))
        return self

    cpdef CsvReaderOptionsBuilder quotechar(self, str quotechar):
        self.c_obj.quotechar(ord(quotechar))
        return self

    cpdef CsvReaderOptionsBuilder decimal(self, str decimal):
        self.c_obj.decimal(ord(decimal))
        return self

    cpdef CsvReaderOptionsBuilder delim_whitespace(self, bool delim_whitespace):
        self.c_obj.delim_whitespace(delim_whitespace)
        return self

    cpdef CsvReaderOptionsBuilder skipinitialspace(self, bool skipinitialspace):
        self.c_obj.skipinitialspace(skipinitialspace)
        return self

    cpdef CsvReaderOptionsBuilder skip_blank_lines(self, bool skip_blank_lines):
        self.c_obj.skip_blank_lines(skip_blank_lines)
        return self

    cpdef CsvReaderOptionsBuilder doublequote(self, bool doublequote):
        self.c_obj.doublequote(doublequote)
        return self

    cpdef CsvReaderOptionsBuilder keep_default_na(self, bool keep_default_na):
        self.c_obj.keep_default_na(keep_default_na)
        return self

    cpdef CsvReaderOptionsBuilder na_filter(self, bool na_filter):
        self.c_obj.na_filter(na_filter)
        return self

    cpdef CsvReaderOptionsBuilder dayfirst(self, bool dayfirst):
        self.c_obj.dayfirst(dayfirst)
        return self

    cpdef CsvReaderOptions build(self):
        cdef CsvReaderOptions csv_options = CsvReaderOptions.__new__(
            CsvReaderOptions
        )
        csv_options.c_obj = move(self.c_obj.build())
        csv_options.source = self.source
        return csv_options


def read_csv(
    CsvReaderOptions options
):
    """Reads a CSV file into a :py:class:`~.types.TableWithMetadata`.

    For details, see :cpp:func:`read_csv`.

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
    prefix : string, default ''
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
    lineterminator : str, default '\\n'
        The character used to determine the end of a line.
    delimiter : str, default ","
        The character used to separate fields in a row.
    thousands : str, default None
        The character used as the thousands separator.
        Cannot match delimiter.
    decimal : str, default '.'
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

    cdef table_with_metadata c_result
    with nogil:
        c_result = move(cpp_read_csv(options.c_obj))

    cdef TableWithMetadata tbl_meta = TableWithMetadata.from_libcudf(c_result)
    return tbl_meta


# TODO: Implement the remaining methods
cdef class CsvWriterOptions:
    """The settings to use for ``write_csv``

    For details, see :cpp:class:`cudf::io::csv_writer_options`
    """
    @staticmethod
    def builder(SinkInfo sink, Table table):
        """Create a CsvWriterOptionsBuilder object

        For details, see :cpp:func:`cudf::io::csv_writer_options::builder`

        Parameters
        ----------
        sink : SinkInfo
            The sink used for writer output
        table : Table
            Table to be written to output

        Returns
        -------
        CsvWriterOptionsBuilder
            Builder to build CsvWriterOptions
        """
        cdef CsvWriterOptionsBuilder csv_builder = CsvWriterOptionsBuilder.__new__(
            CsvWriterOptionsBuilder
        )
        csv_builder.c_obj = csv_writer_options.builder(sink.c_obj, table.view())
        csv_builder.table = table
        csv_builder.sink = sink
        return csv_builder


# TODO: Implement the remaining methods
cdef class CsvWriterOptionsBuilder:
    """Builder to build options for ``write_csv``

    For details, see :cpp:class:`cudf::io::csv_writer_options_builder`
    """
    cpdef CsvWriterOptionsBuilder names(self, list names):
        """Sets optional column names.

        Parameters
        ----------
        names : list[str]
            Column names

        Returns
        -------
        CsvWriterOptionsBuilder
            Builder to build CsvWriterOptions
        """
        self.c_obj.names([name.encode() for name in names])
        return self

    cpdef CsvWriterOptionsBuilder na_rep(self, str val):
        """Sets string to used for null entries.

        Parameters
        ----------
        val : str
            String to represent null value

        Returns
        -------
        CsvWriterOptionsBuilder
            Builder to build CsvWriterOptions
        """
        self.c_obj.na_rep(val.encode())
        return self

    cpdef CsvWriterOptionsBuilder include_header(self, bool val):
        """Enables/Disables headers being written to csv.

        Parameters
        ----------
        val : bool
            Boolean value to enable/disable

        Returns
        -------
        CsvWriterOptionsBuilder
            Builder to build CsvWriterOptions
        """
        self.c_obj.include_header(val)
        return self

    cpdef CsvWriterOptionsBuilder rows_per_chunk(self, int val):
        """Sets maximum number of rows to process for each file write.

        Parameters
        ----------
        val : int
            Number of rows per chunk

        Returns
        -------
        CsvWriterOptionsBuilder
            Builder to build CsvWriterOptions
        """
        self.c_obj.rows_per_chunk(val)
        return self

    cpdef CsvWriterOptionsBuilder line_terminator(self, str term):
        """Sets character used for separating lines.

        Parameters
        ----------
        term : str
            Character to represent line termination

        Returns
        -------
        CsvWriterOptionsBuilder
            Builder to build CsvWriterOptions
        """
        self.c_obj.line_terminator(term.encode())
        return self

    cpdef CsvWriterOptionsBuilder inter_column_delimiter(self, str delim):
        """Sets character used for separating column values.

        Parameters
        ----------
        delim : str
            Character to delimit column values

        Returns
        -------
        CsvWriterOptionsBuilder
            Builder to build CsvWriterOptions
        """
        self.c_obj.inter_column_delimiter(ord(delim))
        return self

    cpdef CsvWriterOptionsBuilder true_value(self, str val):
        """Sets string used for values != 0

        Parameters
        ----------
        val : str
            String to represent values != 0

        Returns
        -------
        CsvWriterOptionsBuilder
            Builder to build CsvWriterOptions
        """
        self.c_obj.true_value(val.encode())
        return self

    cpdef CsvWriterOptionsBuilder false_value(self, str val):
        """Sets string used for values == 0

        Parameters
        ----------
        val : str
            String to represent values == 0

        Returns
        -------
        CsvWriterOptionsBuilder
            Builder to build CsvWriterOptions
        """
        self.c_obj.false_value(val.encode())
        return self

    cpdef CsvWriterOptions build(self):
        """Create a CsvWriterOptions object"""
        cdef CsvWriterOptions csv_options = CsvWriterOptions.__new__(
            CsvWriterOptions
        )
        csv_options.c_obj = move(self.c_obj.build())
        csv_options.table = self.table
        csv_options.sink = self.sink
        return csv_options


cpdef void write_csv(
    CsvWriterOptions options
):
    """
    Write to CSV format.

    The table to write, output paths, and options are encapsulated
    by the `options` object.

    For details, see :cpp:func:`write_csv`.

    Parameters
    ----------
    options: CsvWriterOptions
        Settings for controlling writing behavior
    """

    with nogil:
        cpp_write_csv(move(options.c_obj))
