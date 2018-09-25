# Copyright (c) 2018, NVIDIA CORPORATION.

from __future__ import print_function, division

from libgdf_cffi import libgdf, ffi

from .column import Column
from .numerical import NumericalColumn
from .dataframe import DataFrame
from .datetime import DatetimeColumn


def _wrap_string(text):
    if(text is None):
        return ffi.NULL
    else:
        return ffi.new("char[]", text.encode())


def read_csv(filepath, lineterminator='\n',
             delimiter=',', sep=None, delim_whitespace=False,
             skipinitialspace=False, names=None, dtype=None,
             skipfooter=0, skiprows=0):

    """
    Read CSV data into dataframe.

    Parameters
    ----------
    filepath : str
        Path of file to be read.
    delimiter : char, default ','
        Delimiter to be used.
    delim_whitespace : bool, default False
        Determines whether to use whitespace as delimiter.
    lineterminator : char, default '\n'
        Character to indicate end of line.
    skipinitialspace : bool, default False
        Skip spaces after delimiter.
    names : list of str, default None
        List of column names to be used.
    dtype : list of str, default None
        List of data types for columns.
    skiprows : int, default 0
        Number of rows to be skipped from the start of file.
    skipfooter : int, default 0
        Number of rows to be skipped at the bottom of file.

    Returns
    -------
    GPU ``DataFrame`` object.

    Example
    -------

    foo.txt : ::

        50,50|40,60|30,70|20,80|

    >>> import pygdf
    >>> df = pygdf.read_csv('foo.txt', delimiter=',', lineterminator='|',
    ...                     names=['col1', 'col2'], dtype=['int64', 'int64'],
    ...                     skiprows=1, skipfooter=1)
    >>> df
      col1 col2
    0 40   60
    1 30   70

    """

    csv_reader = ffi.new('csv_read_arg*')

    # Populate csv_reader struct
    file_path = _wrap_string(filepath)
    csv_reader.file_path = file_path
    buffer = _wrap_string(filepath)
    csv_reader.buffer = buffer
    obj = _wrap_string(filepath)
    csv_reader.object = obj

    arr_names = []
    for i, col_name in enumerate(names):
        arr_names.append(_wrap_string(col_name))
    names_ptr = ffi.new('char*[]', arr_names)
    csv_reader.names = names_ptr

    arr_dtypes = []
    for i, col_dtype in enumerate(dtype):
        arr_dtypes.append(_wrap_string(str(col_dtype)))
    dtype_ptr = ffi.new('char*[]', arr_dtypes)
    csv_reader.dtype = dtype_ptr

    csv_reader.delimiter = delimiter.encode()
    csv_reader.lineterminator = lineterminator.encode()
    csv_reader.delim_whitespace = delim_whitespace
    csv_reader.skipinitialspace = skipinitialspace
    x = len(names)
    csv_reader.num_cols = x
    csv_reader.skiprows = skiprows
    csv_reader.skipfooter = skipfooter

    # Call read_csv
    libgdf.read_csv(csv_reader)

    out = csv_reader.data
    if out == ffi.NULL:
        raise ValueError("Failed to parse CSV")

    # Extract parsed columns

    outcols = []
    for i in range(csv_reader.num_cols_out):
        newcol = Column.from_cffi_view(out[i])
        if(newcol.dtype == 'datetime64'):
            outcols.append(newcol.view(DatetimeColumn, dtype='datetime64[s]'))
        else:
            outcols.append(newcol.view(NumericalColumn, dtype=newcol.dtype))

    # Build dataframe
    df = DataFrame()
    for k, v in zip(names, outcols):
        df[k] = v

    return df
