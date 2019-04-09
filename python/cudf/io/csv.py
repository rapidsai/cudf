# Copyright (c) 2018, NVIDIA CORPORATION.

from libgdf_cffi import libgdf, ffi
import nvstrings

from cudf.dataframe.column import Column
from cudf.dataframe.numerical import NumericalColumn
from cudf.dataframe.dataframe import DataFrame
from cudf.dataframe.datetime import DatetimeColumn
from cudf._gdf import nvtx_range_push, nvtx_range_pop

import numpy as np
import collections.abc
import os


def _wrap_string(text):
    if(text is None):
        return ffi.NULL
    else:
        return ffi.new("char[]", text.encode())


def is_file_like(obj):
    if not (hasattr(obj, 'read') or hasattr(obj, 'write')):
        return False
    if not hasattr(obj, "__iter__"):
        return False
    return True


_quoting_enum = {
    0: libgdf.QUOTE_MINIMAL,
    1: libgdf.QUOTE_ALL,
    2: libgdf.QUOTE_NONNUMERIC,
    3: libgdf.QUOTE_NONE,
}


def read_csv(filepath_or_buffer, lineterminator='\n',
             quotechar='"', quoting=0, doublequote=True,
             header='infer',
             mangle_dupe_cols=True, usecols=None,
             sep=',', delimiter=None, delim_whitespace=False,
             skipinitialspace=False, names=None, dtype=None,
             skipfooter=0, skiprows=0, dayfirst=False, compression='infer',
             thousands=None, decimal='.', true_values=None, false_values=None,
             nrows=None, byte_range=None, skip_blank_lines=True, comment=None,
             na_values=None, keep_default_na=True, na_filter=True,
             prefix=None, index_col=None):

    """
    Load and parse a CSV file into a DataFrame

    Parameters
    ----------
    filepath_or_buffer : str
        Path of file to be read or a file-like object containing the file.
    sep : char, default ','
        Delimiter to be used.
    delimiter : char, default None
        Alternative argument name for sep.
    delim_whitespace : bool, default False
        Determines whether to use whitespace as delimiter.
    lineterminator : char, default '\\n'
        Character to indicate end of line.
    skipinitialspace : bool, default False
        Skip spaces after delimiter.
    names : list of str, default None
        List of column names to be used.
    dtype : list of str or dict of {col: dtype}, default None
        List of data types in the same order of the column names
        or a dictionary with column_name:dtype (pandas style).
    quotechar : char, default '"'
        Character to indicate start and end of quote item.
    quoting : str or int, default 0
        Controls quoting behavior. Set to one of
        0 (csv.QUOTE_MINIMAL), 1 (csv.QUOTE_ALL),
        2 (csv.QUOTE_NONNUMERIC) or 3 (csv.QUOTE_NONE).
        Quoting is enabled with all values except 3.
    doublequote : bool, default True
        When quoting is enabled, indicates whether to interpret two
        consecutive quotechar inside fields as single quotechar
    header : int, default 'infer'
        Row number to use as the column names. Default behavior is to infer
        the column names: if no names are passed, header=0;
        if column names are passed explicitly, header=None.
    usecols : list of int or str, default None
        Returns subset of the columns given in the list. All elements must be
        either integer indices (column number) or strings that correspond to
        column names
    mangle_dupe_cols : boolean, default True
        Duplicate columns will be specified as 'X','X.1',...'X.N'.
    skiprows : int, default 0
        Number of rows to be skipped from the start of file.
    skipfooter : int, default 0
        Number of rows to be skipped at the bottom of file.
    compression : {'infer', 'gzip', 'zip', None}, default 'infer'
        For on-the-fly decompression of on-disk data. If ‘infer’, then detect
        compression from the following extensions: ‘.gz’,‘.zip’ (otherwise no
        decompression). If using ‘zip’, the ZIP file must contain only one
        data file to be read in, otherwise the first non-zero-sized file will
        be used. Set to None for no decompression.
    decimal : char, default '.'
        Character used as a decimal point.
    thousands : char, default None
        Character used as a thousands delimiter.
    true_values : list, default None
        Values to consider as boolean True
    false_values : list, default None
        Values to consider as boolean False
    nrows : int, default None
        If specified, maximum number of rows to read
    byte_range : list or tuple, default None
        Byte range within the input file to be read. The first number is the
        offset in bytes, the second number is the range size in bytes. Set the
        size to zero to read all data after the offset location. Reads the row
        that starts before or at the end of the range, even if it ends after
        the end of the range.
    skip_blank_lines : bool, default True
        If True, discard and do not parse empty lines
        If False, interpret empty lines as NaN values
    comment : char, default None
        Character used as a comments indicator. If found at the beginning of a
        line, the line will be ignored altogether.
    na_values : list, default None
        Values to consider as invalid
    keep_default_na : bool, default True
        Whether or not to include the default NA values when parsing the data.
    na_filter : bool, default True
        Detect missing values (empty strings and the values in na_values).
        Passing False can improve performance.
    prefix : str, default None
        Prefix to add to column numbers when parsing without a header row
    index_col : int or string, default None
        Column to use as the row labels

    Returns
    -------
    GPU ``DataFrame`` object.

    Examples
    --------

    Create a test csv file

    >>> import cudf
    >>> filename = 'foo.csv'
    >>> lines = [
    ...   "num1,datetime,text",
    ...   "123,2018-11-13T12:00:00,abc",
    ...   "456,2018-11-14T12:35:01,def",
    ...   "789,2018-11-15T18:02:59,ghi"
    ... ]
    >>> with open(filename, 'w') as fp:
    ...     fp.write('\\n'.join(lines)+'\\n')

    Read the file with ``cudf.read_csv``

    >>> cudf.read_csv(filename)
      num1                datetime text
    0  123 2018-11-13T12:00:00.000 5451
    1  456 2018-11-14T12:35:01.000 5784
    2  789 2018-11-15T18:02:59.000 6117
    """

    if delim_whitespace:
        if delimiter is not None:
            raise ValueError("cannot set both delimiter and delim_whitespace")
        if sep != ',':
            raise ValueError("cannot set both sep and delim_whitespace")

    # Alias sep -> delimiter.
    if delimiter is None:
        delimiter = sep

    if dtype is not None:
        if isinstance(dtype, collections.abc.Mapping):
            dtype_dict = True
        elif isinstance(dtype, collections.abc.Iterable):
            dtype_dict = False
        else:
            msg = '''dtype must be 'list like' or 'dict' '''
            raise TypeError(msg)
        if names is not None and len(dtype) != len(names):
            msg = '''All column dtypes must be specified.'''
            raise TypeError(msg)

    nvtx_range_push("CUDF_READ_CSV", "purple")

    csv_reader = ffi.new('csv_read_arg*')

    # Populate csv_reader struct
    if is_file_like(filepath_or_buffer):
        if compression == 'infer':
            compression = None
        buffer = filepath_or_buffer.read()
        # check if StringIO is used
        if hasattr(buffer, 'encode'):
            buffer_as_bytes = buffer.encode()
        else:
            buffer_as_bytes = buffer
        buffer_data_holder = ffi.new("char[]", buffer_as_bytes)

        csv_reader.input_data_form = libgdf.HOST_BUFFER
        csv_reader.filepath_or_buffer = buffer_data_holder
        csv_reader.buffer_size = len(buffer_as_bytes)
    else:
        if (not os.path.isfile(filepath_or_buffer)):
            raise(FileNotFoundError)
        if (not os.path.exists(filepath_or_buffer)):
            raise(FileNotFoundError)
        file_path = _wrap_string(filepath_or_buffer)

        csv_reader.input_data_form = libgdf.FILE_PATH
        csv_reader.filepath_or_buffer = file_path

    if header == 'infer':
        header = -1
    header_infer = header
    arr_names = []
    arr_dtypes = []
    if names is None:
        if header is -1:
            header_infer = 0
        if header is None:
            header_infer = -1
        csv_reader.names = ffi.NULL
        csv_reader.num_cols = 0
        if dtype is not None:
            if dtype_dict:
                for k, v in dtype.items():
                    arr_dtypes.append(_wrap_string(str(str(k)+":"+str(v))))
    else:
        if header is None:
            header_infer = -1
        csv_reader.num_cols = len(names)
        for col_name in names:
            arr_names.append(_wrap_string(col_name))
            if dtype is not None:
                if dtype_dict:
                    arr_dtypes.append(_wrap_string(str(dtype[col_name])))
        names_ptr = ffi.new('char*[]', arr_names)
        csv_reader.names = names_ptr

    if dtype is None:
        csv_reader.dtype = ffi.NULL
    else:
        if not dtype_dict:
            for col_dtype in dtype:
                arr_dtypes.append(_wrap_string(str(col_dtype)))
        dtype_ptr = ffi.new('char*[]', arr_dtypes)
        csv_reader.dtype = dtype_ptr

    csv_reader.use_cols_int = ffi.NULL
    csv_reader.use_cols_int_len = 0
    csv_reader.use_cols_char = ffi.NULL
    csv_reader.use_cols_char_len = 0

    if usecols is not None:
        arr_col_names = []
        if(all(isinstance(x, int) for x in usecols)):
            usecols_ptr = ffi.new('int[]', usecols)
            csv_reader.use_cols_int = usecols_ptr
            csv_reader.use_cols_int_len = len(usecols)
        else:
            for col_name in usecols:
                arr_col_names.append(_wrap_string(col_name))
            col_names_ptr = ffi.new('char*[]', arr_col_names)
            csv_reader.use_cols_char = col_names_ptr
            csv_reader.use_cols_char_len = len(usecols)

    if decimal == delimiter:
        raise ValueError("decimal cannot be the same as delimiter")

    if thousands == delimiter:
        raise ValueError("thousands cannot be the same as delimiter")

    if nrows is not None and skipfooter != 0:
        raise ValueError("cannot use both nrows and skipfooter parameters")

    if byte_range is not None:
        if skipfooter != 0 or skiprows != 0 or nrows is not None:
            raise ValueError("""cannot manually limit rows to be read when
                                using the byte range parameter""")

    arr_true_values = []
    for value in true_values or []:
        arr_true_values.append(_wrap_string(str(value)))
    arr_true_values_ptr = ffi.new('char*[]', arr_true_values)
    csv_reader.true_values = arr_true_values_ptr
    csv_reader.num_true_values = len(arr_true_values)

    arr_false_values = []
    for value in false_values or []:
        arr_false_values.append(_wrap_string(str(value)))
    false_values_ptr = ffi.new('char*[]', arr_false_values)
    csv_reader.false_values = false_values_ptr
    csv_reader.num_false_values = len(arr_false_values)

    arr_na_values = []
    for value in na_values or []:
        arr_na_values.append(_wrap_string(str(value)))
    arr_na_values_ptr = ffi.new('char*[]', arr_na_values)
    csv_reader.na_values = arr_na_values_ptr
    csv_reader.num_na_values = len(arr_na_values)

    compression_bytes = _wrap_string(compression)
    prefix_bytes = _wrap_string(prefix)

    csv_reader.delimiter = delimiter.encode()
    csv_reader.lineterminator = lineterminator.encode()
    csv_reader.quotechar = quotechar.encode()
    csv_reader.quoting = _quoting_enum[quoting]
    csv_reader.doublequote = doublequote
    csv_reader.delim_whitespace = delim_whitespace
    csv_reader.skipinitialspace = skipinitialspace
    csv_reader.dayfirst = dayfirst
    csv_reader.header = header_infer
    csv_reader.skiprows = skiprows
    csv_reader.skipfooter = skipfooter
    csv_reader.mangle_dupe_cols = mangle_dupe_cols
    csv_reader.windowslinetermination = False
    csv_reader.compression = compression_bytes
    csv_reader.decimal = decimal.encode()
    csv_reader.thousands = thousands.encode() if thousands else b'\0'
    csv_reader.nrows = nrows if nrows is not None else -1
    if byte_range is not None:
        csv_reader.byte_range_offset = byte_range[0]
        csv_reader.byte_range_size = byte_range[1]
    else:
        csv_reader.byte_range_offset = 0
        csv_reader.byte_range_size = 0
    csv_reader.skip_blank_lines = skip_blank_lines
    csv_reader.comment = comment.encode() if comment else b'\0'
    csv_reader.keep_default_na = keep_default_na
    csv_reader.na_filter = na_filter
    csv_reader.prefix = prefix_bytes

    # Call read_csv
    libgdf.read_csv(csv_reader)

    out = csv_reader.data
    if out == ffi.NULL:
        raise ValueError("Failed to parse CSV")

    # Extract parsed columns

    outcols = []
    new_names = []
    for i in range(csv_reader.num_cols_out):
        if out[i].dtype == libgdf.GDF_STRING:
            ptr = int(ffi.cast("uintptr_t", out[i].data))
            new_names.append(ffi.string(out[i].col_name).decode())
            outcols.append(nvstrings.bind_cpointer(ptr))
        else:
            newcol = Column.from_cffi_view(out[i])
            new_names.append(ffi.string(out[i].col_name).decode())
            if(newcol.dtype.type == np.datetime64):
                outcols.append(
                    newcol.view(DatetimeColumn, dtype='datetime64[ms]')
                )
            else:
                outcols.append(
                    newcol.view(NumericalColumn, dtype=newcol.dtype)
                )

    # Build dataframe
    df = DataFrame()
    # if names is not None and header_infer is -1:

    for k, v in zip(new_names, outcols):
        df[k] = v

    # Set index if the index_col parameter is passed
    if index_col is not None and index_col is not False:
        if isinstance(index_col, (int)):
            df = df.set_index(df.columns[index_col])
        else:
            df = df.set_index(index_col)

    nvtx_range_pop()

    return df
