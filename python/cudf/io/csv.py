# Copyright (c) 2018, NVIDIA CORPORATION.

from cudf.bindings.csv import cpp_read_csv


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

    return cpp_read_csv(
        filepath_or_buffer,
        lineterminator=lineterminator,
        quotechar=quotechar,
        quoting=quoting,
        doublequote=doublequote,
        header=header,
        mangle_dupe_cols=mangle_dupe_cols,
        usecols=usecols,
        sep=sep,
        delimiter=delimiter,
        delim_whitespace=delim_whitespace,
        skipinitialspace=skipinitialspace,
        names=names,
        dtype=dtype,
        skipfooter=skipfooter,
        skiprows=skiprows,
        dayfirst=dayfirst,
        compression=compression,
        thousands=thousands,
        decimal=decimal,
        true_values=true_values,
        false_values=false_values,
        nrows=nrows,
        byte_range=byte_range,
        skip_blank_lines=skip_blank_lines,
        comment=comment,
        na_values=na_values,
        keep_default_na=keep_default_na,
        na_filter=na_filter,
        prefix=prefix,
        index_col=index_col
    )
