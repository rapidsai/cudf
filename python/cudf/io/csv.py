# Copyright (c) 2018, NVIDIA CORPORATION.

from libgdf_cffi import libgdf, ffi

from cudf.dataframe.dataframe import Column
from cudf.dataframe.numerical import NumericalColumn
from cudf.dataframe.dataframe import DataFrame
from cudf.dataframe.datetime import DatetimeColumn
from cudf._gdf import nvtx_range_push, nvtx_range_pop

import numpy as np
import collections.abc


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


def read_csv(filepath_or_buffer, lineterminator='\n',
             quotechar='"', quoting=True, doublequote=True,
             header='infer',
             mangle_dupe_cols=True, usecols=None,
             delimiter=',', sep=None, delim_whitespace=False,
             skipinitialspace=False, names=None, dtype=None,
             skipfooter=0, skiprows=0, dayfirst=False, compression='infer',
             thousands=None, decimal='.', true_values=None, false_values=None,
             nrows=None):
    """
    Load and parse a CSV file into a DataFrame

    Parameters
    ----------
    filepath_or_buffer : str
        Path of file to be read or a file-like object containing the file.
    delimiter : char, default ','
        Delimiter to be used.
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
    quoting : bool, default True
        If True, start and end quotechar are removed from returned strings
        If False, start and end quotechar are kept in returned strings
    doublequote : bool, default True
        When quotechar is specified and quoting is True, indicates whether to
        interpret two consecutive quotechar inside fields as single quotechar
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
    nrows: int, default None
        If specified, maximum number of rows to read

    Returns
    -------
    GPU ``DataFrame`` object.

    Examples
    --------


    .. code-block:: python

      import cudf

      # Create a test csv file
      filename = 'foo.csv'
      lines = [
        "num1,datetime,text",
        "123,2018-11-13T12:00:00,abc",
        "456,2018-11-14T12:35:01,def",
        "789,2018-11-15T18:02:59,ghi"
      ]
      with open(filename, 'w') as fp:
          fp.write('\\n'.join(lines)+'\\n')

      # Read the file with cudf
      names = ['num1', 'datetime', 'text']
      # Note 'int' for 3rd column- text will be hashed
      dtypes = ['int', 'date', 'int']
      df = cudf.read_csv(filename, delimiter=',',
                         names=names, dtype=dtypes,
                         skiprows=1)

      # Display results
      print(df)

    Output:

    .. code-block:: python

          num1                datetime text
        0  123 2018-11-13T12:00:00.000 5451
        1  456 2018-11-14T12:35:01.000 5784
        2  789 2018-11-15T18:02:59.000 6117

    See Also
    --------
    .read_csv_strings
    """
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

    nvtx_range_push("PYGDF_READ_CSV", "purple")

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
        file_path = _wrap_string(filepath_or_buffer)

        csv_reader.input_data_form = libgdf.FILE_PATH
        csv_reader.filepath_or_buffer = file_path

    if header is 'infer':
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

    # Start with default values recognized as boolean
    arr_true_values = [_wrap_string(str('True')), _wrap_string(str('TRUE'))]
    arr_false_values = [_wrap_string(str('False')), _wrap_string(str('FALSE'))]

    for value in true_values or []:
        arr_true_values.append(_wrap_string(str(value)))
    arr_true_values_ptr = ffi.new('char*[]', arr_true_values)
    csv_reader.true_values = arr_true_values_ptr
    csv_reader.num_true_values = len(arr_true_values)

    for value in false_values or []:
        arr_false_values.append(_wrap_string(str(value)))
    false_values_ptr = ffi.new('char*[]', arr_false_values)
    csv_reader.false_values = false_values_ptr
    csv_reader.num_false_values = len(arr_false_values)

    compression_bytes = _wrap_string(compression)

    csv_reader.delimiter = delimiter.encode()
    csv_reader.lineterminator = lineterminator.encode()
    csv_reader.quotechar = quotechar.encode()
    csv_reader.quoting = quoting
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

    # Call read_csv
    libgdf.read_csv(csv_reader)

    out = csv_reader.data
    if out == ffi.NULL:
        raise ValueError("Failed to parse CSV")

    # Extract parsed columns

    outcols = []
    new_names = []
    for i in range(csv_reader.num_cols_out):
        newcol = Column.from_cffi_view(out[i])
        new_names.append(ffi.string(out[i].col_name).decode())
        if(newcol.dtype == np.dtype('datetime64[ms]')):
            outcols.append(newcol.view(DatetimeColumn, dtype='datetime64[ms]'))
        else:
            outcols.append(newcol.view(NumericalColumn, dtype=newcol.dtype))

    # Build dataframe
    df = DataFrame()
    # if names is not None and header_infer is -1:

    for k, v in zip(new_names, outcols):
        df[k] = v

    nvtx_range_pop()

    return df


def read_csv_strings(filepath_or_buffer, lineterminator='\n',
                     quotechar='"', quoting=True, doublequote=True,
                     delimiter=',', sep=None, delim_whitespace=False,
                     skipinitialspace=False, names=None, dtype=None,
                     skipfooter=0, skiprows=0, dayfirst=False,
                     compression='infer', thousands=None, decimal='.',
                     true_values=None, false_values=None, nrows=None):

    """
    **Experimental**: This function exists only as a beta way to use
    `nvstrings <https://nvstrings.readthedocs.io/en/latest/>`_. with cudf.

    Future versions of cuDF will provide cleaner integration.

    Uses mostly same arguments as read_csv.
    Note: Doesn't currently support auto-column detection, header, usecols
    and mangle_dupe_cols args.

    Returns
    -------
    columns : ordered list of cudf.dataframe.Series and nvstrings objects
      numeric or date dtyped columns will be Series.

      'str' dtyped columns will be
      `nvstrings <https://nvstrings.readthedocs.io/en/latest/>`_.

    Examples
    --------

    .. code-block:: python

      import cudf

      # Create a test csv file
      filename = 'foo.csv'
      lines = [
        "num1,datetime,text",
        "123,2018-11-13T12:00:00,abc",
        "456,2018-11-14T12:35:01,def",
        "789,2018-11-15T18:02:59,ghi"
      ]
      with open(filename, 'w') as fp:
          fp.write('\\n'.join(lines)+'\\n')

      # Read the file with cudf
      names = ['num1', 'datetime', 'text']
      dtypes = ['int', 'date', 'str']
      columns = cudf.io.csv.read_csv_strings(filename, delimiter=',',
                              names=names, dtype=dtypes,
                              skiprows=1)
      # Display results
      columns[0]
      print(columns[0])
      columns[2]
      print(columns[2])

    Output:

    .. code-block:: python

      <cudf.Series nrows=3 >
      0  123
      1  456
      2  789

      <nvstrings count=3>
      ['abc', 'def', 'ghi']

    See Also
    --------
    .read_csv
    """
    import nvstrings
    from cudf.dataframe.series import Series

    if names is None or dtype is None:
        msg = '''Automatic dtype detection not implemented:
        Column names and dtypes must be specified.'''
        raise TypeError(msg)

    if isinstance(dtype, dict):
        dtype_dict = True
    elif isinstance(dtype, list):
        dtype_dict = False
        if len(dtype) != len(names):
            msg = '''All column dtypes must be specified.'''
            raise TypeError(msg)
    else:
        msg = '''dtype must be 'list' or 'dict' '''
        raise TypeError(msg)

    csv_reader = ffi.new('csv_read_arg*')

    # Populate csv_reader struct
    if is_file_like(filepath_or_buffer):
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
        file_path = _wrap_string(filepath_or_buffer)

        csv_reader.input_data_form = libgdf.FILE_PATH
        csv_reader.filepath_or_buffer = file_path

    arr_names = []
    arr_dtypes = []
    for col_name in names:
        arr_names.append(_wrap_string(col_name))
        if dtype_dict:
            arr_dtypes.append(_wrap_string(str(dtype[col_name])))
    names_ptr = ffi.new('char*[]', arr_names)
    csv_reader.names = names_ptr

    if not dtype_dict:
        for col_dtype in dtype:
            arr_dtypes.append(_wrap_string(str(col_dtype)))
    dtype_ptr = ffi.new('char*[]', arr_dtypes)
    csv_reader.dtype = dtype_ptr

    if decimal == delimiter:
        raise ValueError("decimal cannot be the same as delimiter")

    if thousands == delimiter:
        raise ValueError("thousands cannot be the same as delimiter")

    if nrows is not None and skipfooter != 0:
        raise ValueError("cannot use both nrows and skipfooter parameters")

    # Start with default values recognized as boolean
    arr_true_values = [_wrap_string(str('True')), _wrap_string(str('TRUE'))]
    arr_false_values = [_wrap_string(str('False')), _wrap_string(str('FALSE'))]

    for value in true_values or []:
        arr_true_values.append(_wrap_string(str(value)))
    arr_true_values_ptr = ffi.new('char*[]', arr_true_values)
    csv_reader.true_values = arr_true_values_ptr
    csv_reader.num_true_values = len(arr_true_values)

    for value in false_values or []:
        arr_false_values.append(_wrap_string(str(value)))
    false_values_ptr = ffi.new('char*[]', arr_false_values)
    csv_reader.false_values = false_values_ptr
    csv_reader.num_false_values = len(arr_false_values)

    compression_bytes = _wrap_string(compression)

    csv_reader.delimiter = delimiter.encode()
    csv_reader.lineterminator = lineterminator.encode()
    csv_reader.quotechar = quotechar.encode()
    csv_reader.quoting = quoting
    csv_reader.doublequote = doublequote
    csv_reader.delim_whitespace = delim_whitespace
    csv_reader.skipinitialspace = skipinitialspace
    csv_reader.dayfirst = dayfirst
    csv_reader.num_cols = len(names)
    csv_reader.skiprows = skiprows
    csv_reader.skipfooter = skipfooter
    csv_reader.compression = compression_bytes
    csv_reader.decimal = decimal.encode()
    csv_reader.thousands = thousands.encode() if thousands else b'\0'
    csv_reader.nrows = nrows if nrows is not None else -1

    # Call read_csv
    libgdf.read_csv(csv_reader)

    out = csv_reader.data
    if out == ffi.NULL:
        raise ValueError("Failed to parse CSV")

    # Extract parsed columns

    outcols = []
    for i in range(csv_reader.num_cols_out):
        if out[i].dtype == libgdf.GDF_STRING:
            ptr = int(ffi.cast("uintptr_t", out[i].data))
            outcols.append(nvstrings.bind_cpointer(ptr))
        else:
            newcol = Column.from_cffi_view(out[i])
            if(newcol.dtype == np.dtype('datetime64[ms]')):
                col = newcol.view(DatetimeColumn, dtype='datetime64[ms]')
            else:
                col = newcol.view(NumericalColumn, dtype=newcol.dtype)
            outcols.append(Series(col))

    return outcols
