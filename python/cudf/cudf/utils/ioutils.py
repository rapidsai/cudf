# Copyright (c) 2019, NVIDIA CORPORATION.

import os
import urllib
import warnings
from io import BytesIO, TextIOWrapper

import fsspec

from cudf.utils.docutils import docfmt_partial

_docstring_remote_sources = """
- cuDF supports local and remote data stores. See configuration details for
  available sources
  `here <https://docs.dask.org/en/latest/remote-data-services.html>`_.
"""

_docstring_read_avro = """
Load an Avro dataset into a DataFrame

Parameters
----------
filepath_or_buffer : str, path object, bytes, or file-like object
    Either a path to a file (a `str`, `pathlib.Path`, or
    `py._path.local.LocalPath`), URL (including http, ftp, and S3 locations),
    Python bytes of raw binary data, or any object with a `read()` method
    (such as builtin `open()` file handler function or `BytesIO`).
engine : {{ 'cudf', 'fastavro' }}, default 'cudf'
    Parser engine to use.
columns : list, default None
    If not None, only these columns will be read.
skip_rows : int, default None
    If not None, the nunber of rows to skip from the start of the file.
num_rows : int, default None
    If not None, the total number of rows to read.

Returns
-------
DataFrame

Notes
-----
{remote_data_sources}

Examples
--------
>>> import cudf
>>> df = cudf.read_avro(filename)
>>> df
  num1                datetime text
0  123 2018-11-13T12:00:00.000 5451
1  456 2018-11-14T12:35:01.000 5784
2  789 2018-11-15T18:02:59.000 6117

See Also
--------
cudf.io.csv.read_csv
cudf.io.json.read_json
""".format(
    remote_data_sources=_docstring_remote_sources
)
doc_read_avro = docfmt_partial(docstring=_docstring_read_avro)

_docstring_read_parquet_metadata = """
Read a Parquet file's metadata and schema

Parameters
----------
path : string or path object
    Path of file to be read

Returns
-------
Total number of rows
Number of row groups
List of column names

Examples
--------
>>> import cudf
>>> num_rows, num_row_groups, names = cudf.io.read_parquet_metadata(filename)
>>> df = [cudf.read_parquet(fname, row_group=i) for i in range(row_groups)]
>>> df = cudf.concat(df)
>>> df
  num1                datetime text
0  123 2018-11-13T12:00:00.000 5451
1  456 2018-11-14T12:35:01.000 5784
2  789 2018-11-15T18:02:59.000 6117

See Also
--------
cudf.io.parquet.read_parquet
"""
doc_read_parquet_metadata = docfmt_partial(
    docstring=_docstring_read_parquet_metadata
)

_docstring_read_parquet = """
Load a Parquet dataset into a DataFrame

Parameters
----------
filepath_or_buffer : str, path object, bytes, or file-like object
    Either a path to a file (a `str`, `pathlib.Path`, or
    `py._path.local.LocalPath`), URL (including http, ftp, and S3 locations),
    Python bytes of raw binary data, or any object with a `read()` method
    (such as builtin `open()` file handler function or `BytesIO`).
engine : {{ 'cudf', 'pyarrow' }}, default 'cudf'
    Parser engine to use.
columns : list, default None
    If not None, only these columns will be read.
row_group : int, default None
    If not None, only the row group with the specified index will be read.
skip_rows : int, default None
    If not None, the nunber of rows to skip from the start of the file.
num_rows : int, default None
    If not None, the total number of rows to read.
strings_to_categorical : boolean, default False
    If True, return string columns as GDF_CATEGORY dtype; if False, return a
    as GDF_STRING dtype.
use_pandas_metadata : boolean, default True
    If True and dataset has custom PANDAS schema metadata, ensure that index
    columns are also loaded.

Returns
-------
DataFrame

Notes
-----
{remote_data_sources}

Examples
--------
>>> import cudf
>>> df = cudf.read_parquet(filename)
>>> df
  num1                datetime text
0  123 2018-11-13T12:00:00.000 5451
1  456 2018-11-14T12:35:01.000 5784
2  789 2018-11-15T18:02:59.000 6117

See Also
--------
cudf.io.parquet.read_parquet_metadata
cudf.io.parquet.to_parquet
cudf.io.orc.read_orc
""".format(
    remote_data_sources=_docstring_remote_sources
)
doc_read_parquet = docfmt_partial(docstring=_docstring_read_parquet)

_docstring_to_parquet = """
Write a DataFrame to the parquet format.

Parameters
----------
path : str
    File path or Root Directory path. Will be used as Root Directory path
    while writing a partitioned dataset.
compression : {'snappy', 'gzip', 'brotli', None}, default 'snappy'
    Name of the compression to use. Use ``None`` for no compression.
index : bool, default None
    If ``True``, include the dataframe's index(es) in the file output. If
    ``False``, they will not be written to the file. If ``None``, the
    engine's default behavior will be used.
partition_cols : list, optional, default None
    Column names by which to partition the dataset
    Columns are partitioned in the order they are given

See Also
--------
cudf.io.parquet.read_parquet
cudf.io.orc.read_orc
"""
doc_to_parquet = docfmt_partial(docstring=_docstring_to_parquet)

_docstring_read_orc_metadata = """
Read an ORC file's metadata and schema

Parameters
----------
path : string or path object
    Path of file to be read

Returns
-------
Total number of rows
Number of stripes
List of column names

Examples
--------
>>> import cudf
>>> num_rows, stripes, names = cudf.io.read_orc_metadata(filename)
>>> df = [cudf.read_orc(fname, stripe=i) for i in range(stripes)]
>>> df = cudf.concat(df)
>>> df
  num1                datetime text
0  123 2018-11-13T12:00:00.000 5451
1  456 2018-11-14T12:35:01.000 5784
2  789 2018-11-15T18:02:59.000 6117

See Also
--------
cudf.io.orc.read_orc
"""
doc_read_orc_metadata = docfmt_partial(docstring=_docstring_read_orc_metadata)

_docstring_read_orc = """
Load an ORC dataset into a DataFrame

Parameters
----------
filepath_or_buffer : str, path object, bytes, or file-like object
    Either a path to a file (a `str`, `pathlib.Path`, or
    `py._path.local.LocalPath`), URL (including http, ftp, and S3 locations),
    Python bytes of raw binary data, or any object with a `read()` method
    (such as builtin `open()` file handler function or `BytesIO`).
engine : {{ 'cudf', 'pyarrow' }}, default 'cudf'
    Parser engine to use.
columns : list, default None
    If not None, only these columns will be read from the file.
stripe: int, default None
    If not None, only the stripe with the specified index will be read.
skip_rows : int, default None
    If not None, the number of rows to skip from the start of the file.
num_rows : int, default None
    If not None, the total number of rows to read.
use_index : bool, default True
    If True, use row index if available for faster seeking.
kwargs are passed to the engine

Returns
-------
DataFrame

Notes
-----
{remote_data_sources}

Examples
--------
>>> import cudf
>>> df = cudf.read_orc(filename)
>>> df
  num1                datetime text
0  123 2018-11-13T12:00:00.000 5451
1  456 2018-11-14T12:35:01.000 5784
2  789 2018-11-15T18:02:59.000 6117

See Also
--------
cudf.io.parquet.read_parquet
cudf.io.parquet.to_parquet
""".format(
    remote_data_sources=_docstring_remote_sources
)
doc_read_orc = docfmt_partial(docstring=_docstring_read_orc)

_docstring_to_orc = """
Write a DataFrame to the ORC format.

Parameters
----------
fname : str
    File path or object where the ORC dataset will be stored.
compression : {{ 'snappy', None }}, default None
    Name of the compression to use. Use None for no compression.

See Also
--------
cudf.io.orc.read_orc
"""
doc_to_orc = docfmt_partial(docstring=_docstring_to_orc)

_docstring_read_json = """
Load a JSON dataset into a DataFrame

Parameters
----------
path_or_buf : str, path object, or file-like object
    Either JSON data in a `str`, path to a file (a `str`, `pathlib.Path`, or
    `py._path.local.LocalPath`), URL (including http, ftp, and S3 locations),
    or any object with a `read()` method (such as builtin `open()` file handler
    function or `StringIO`).
engine : {{ 'auto', 'cudf', 'pandas' }}, default 'auto'
    Parser engine to use. If 'auto' is passed, the engine will be
    automatically selected based on the other parameters.
orient : string,
    Indication of expected JSON string format (pandas engine only).
    Compatible JSON strings can be produced by ``to_json()`` with a
    corresponding orient value.
    The set of possible orients is:

    - ``'split'`` : dict like
      ``{index -> [index], columns -> [columns], data -> [values]}``
    - ``'records'`` : list like
      ``[{column -> value}, ... , {column -> value}]``
    - ``'index'`` : dict like ``{index -> {column -> value}}``
    - ``'columns'`` : dict like ``{column -> {index -> value}}``
    - ``'values'`` : just the values array

    The allowed and default values depend on the value
    of the `typ` parameter.

    * when ``typ == 'series'``,

      - allowed orients are ``{'split','records','index'}``
      - default is ``'index'``
      - The Series index must be unique for orient ``'index'``.
    * when ``typ == 'frame'``,

      - allowed orients are ``{'split','records','index',
        'columns','values', 'table'}``
      - default is ``'columns'``
      - The DataFrame index must be unique for orients ``'index'`` and
        ``'columns'``.
      - The DataFrame columns must be unique for orients ``'index'``,
        ``'columns'``, and ``'records'``.
typ : type of object to recover (series or frame), default 'frame'
    With cudf engine, only frame output is supported.
dtype : boolean or dict, default True
    If True, infer dtypes, if a dict of column to dtype, then use those,
    if False, then don't infer dtypes at all, applies only to the data.
convert_axes : boolean, default True
    Try to convert the axes to the proper dtypes (pandas engine only).
convert_dates : boolean, default True
    List of columns to parse for dates (pandas engine only); If True, then try
    to parse datelike columns default is True; a column label is datelike if

    * it ends with ``'_at'``,
    * it ends with ``'_time'``,
    * it begins with ``'timestamp'``,
    * it is ``'modified'``, or
    * it is ``'date'``
keep_default_dates : boolean, default True
    If parsing dates, parse the default datelike columns (pandas engine only)
numpy : boolean, default False
    Direct decoding to numpy arrays (pandas engine only). Supports numeric
    data only, but non-numeric column and index labels are supported. Note
    also that the JSON ordering MUST be the same for each term if numpy=True.
precise_float : boolean, default False
    Set to enable usage of higher precision (strtod) function when
    decoding string to double values (pandas engine only). Default (False)
    is to use fast but less precise builtin functionality
date_unit : string, default None
    The timestamp unit to detect if converting dates (pandas engine only).
    The default behaviour is to try and detect the correct precision, but if
    this is not desired then pass one of 's', 'ms', 'us' or 'ns' to force
    parsing only seconds, milliseconds, microseconds or nanoseconds.
encoding : str, default is 'utf-8'
    The encoding to use to decode py3 bytes.
    With cudf engine, only utf-8 is supported.
lines : boolean, default False
    Read the file as a json object per line.
chunksize : integer, default None
    Return JsonReader object for iteration (pandas engine only).
    See the `line-delimted json docs
    <http://pandas.pydata.org/pandas-docs/stable/io.html#io-jsonl>`_
    for more information on ``chunksize``.
    This can only be passed if `lines=True`.
    If this is None, the file will be read into memory all at once.
compression : {'infer', 'gzip', 'bz2', 'zip', 'xz', None}, default 'infer'
    For on-the-fly decompression of on-disk data. If 'infer', then use
    gzip, bz2, zip or xz if path_or_buf is a string ending in
    '.gz', '.bz2', '.zip', or 'xz', respectively, and no decompression
    otherwise. If using 'zip', the ZIP file must contain only one data
    file to be read in. Set to None for no decompression.
byte_range : list or tuple, default None
    Byte range within the input file to be read (cudf engine only).
    The first number is the offset in bytes, the second number is the range
    size in bytes. Set the size to zero to read all data after the offset
    location. Reads the row that starts before or at the end of the range,
    even if it ends after the end of the range.

Returns
-------
result : Series or DataFrame, depending on the value of `typ`.

See Also
--------
.cudf.io.json.to_json
"""
doc_read_json = docfmt_partial(docstring=_docstring_read_json)

_docstring_to_json = """
Convert the cuDF object to a JSON string.
Note nulls and NaNs will be converted to null and datetime objects
will be converted to UNIX timestamps.

Parameters
----------
path_or_buf : string or file handle, optional
    File path or object. If not specified, the result is returned as a string.
orient : string
    Indication of expected JSON string format.

    * Series
        - default is 'index'
        - allowed values are: {'split','records','index','table'}
    * DataFrame
        - default is 'columns'
        - allowed values are:
          {'split','records','index','columns','values','table'}
    * The format of the JSON string
        - 'split' : dict like {'index' -> [index],
          'columns' -> [columns], 'data' -> [values]}
        - 'records' : list like
          [{column -> value}, ... , {column -> value}]
        - 'index' : dict like {index -> {column -> value}}
        - 'columns' : dict like {column -> {index -> value}}
        - 'values' : just the values array
        - 'table' : dict like {'schema': {schema}, 'data': {data}}
          describing the data, and the data component is
          like ``orient='records'``.
date_format : {None, 'epoch', 'iso'}
    Type of date conversion. 'epoch' = epoch milliseconds,
    'iso' = ISO8601. The default depends on the `orient`. For
    ``orient='table'``, the default is 'iso'. For all other orients,
    the default is 'epoch'.
double_precision : int, default 10
    The number of decimal places to use when encoding
    floating point values.
force_ascii : bool, default True
    Force encoded string to be ASCII.
date_unit : string, default 'ms' (milliseconds)
    The time unit to encode to, governs timestamp and ISO8601
    precision.  One of 's', 'ms', 'us', 'ns' for second, millisecond,
    microsecond, and nanosecond respectively.
default_handler : callable, default None
    Handler to call if object cannot otherwise be converted to a
    suitable format for JSON. Should receive a single argument which is
    the object to convert and return a serialisable object.
lines : bool, default False
    If 'orient' is 'records' write out line delimited json format. Will
    throw ValueError if incorrect 'orient' since others are not list
    like.
compression : {'infer', 'gzip', 'bz2', 'zip', 'xz', None}
    A string representing the compression to use in the output file,
    only used when the first argument is a filename. By default, the
    compression is inferred from the filename.
index : bool, default True
    Whether to include the index values in the JSON string. Not
    including the index (``index=False``) is only supported when
    orient is 'split' or 'table'.

See Also
--------
.cudf.io.json.read_json
"""
doc_to_json = docfmt_partial(docstring=_docstring_to_json)

_docstring_read_hdf = """
Read from the store, close it if we opened it.

Retrieve pandas object stored in file, optionally based on where
criteria

Parameters
----------
path_or_buf : string, buffer or path object
    Path to the file to open, or an open `HDFStore
    <https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html#hdf5-pytables>`_.
    object.
    Supports any object implementing the ``__fspath__`` protocol.
    This includes :class:`pathlib.Path` and py._path.local.LocalPath
    objects.
key : object, optional
    The group identifier in the store. Can be omitted if the HDF file
    contains a single pandas object.
mode : {'r', 'r+', 'a'}, optional
    Mode to use when opening the file. Ignored if path_or_buf is a
    `Pandas HDFS
    <https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html#hdf5-pytables>`_.
    Default is 'r'.
where : list, optional
    A list of Term (or convertible) objects.
start : int, optional
    Row number to start selection.
stop  : int, optional
    Row number to stop selection.
columns : list, optional
    A list of columns names to return.
iterator : bool, optional
    Return an iterator object.
chunksize : int, optional
    Number of rows to include in an iteration when using an iterator.
errors : str, default 'strict'
    Specifies how encoding and decoding errors are to be handled.
    See the errors argument for :func:`open` for a full list
    of options.
**kwargs
    Additional keyword arguments passed to HDFStore.

Returns
-------
item : object
    The selected object. Return type depends on the object stored.

See Also
--------
cudf.io.hdf.to_hdf : Write a HDF file from a DataFrame.
"""
doc_read_hdf = docfmt_partial(docstring=_docstring_read_hdf)

_docstring_to_hdf = """
Write the contained data to an HDF5 file using HDFStore.

Hierarchical Data Format (HDF) is self-describing, allowing an
application to interpret the structure and contents of a file with
no outside information. One HDF file can hold a mix of related objects
which can be accessed as a group or as individual objects.

In order to add another DataFrame or Series to an existing HDF file
please use append mode and a different a key.

For more information see the `user guide
<https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html#hdf5-pytables>`_.

Parameters
----------
path_or_buf : str or pandas.HDFStore
    File path or HDFStore object.
key : str
    Identifier for the group in the store.
mode : {'a', 'w', 'r+'}, default 'a'
    Mode to open file:

    - 'w': write, a new file is created (an existing file with the same name
      would be deleted).
    - 'a': append, an existing file is opened for reading and writing, and if
      the file does not exist it is created.
    - 'r+': similar to 'a', but the file must already exist.
format : {'fixed', 'table'}, default 'fixed'
    Possible values:

    - 'fixed': Fixed format. Fast writing/reading. Not-appendable,
      nor searchable.
    - 'table': Table format. Write as a PyTables Table structure
      which may perform worse but allow more flexible operations
      like searching / selecting subsets of the data.
append : bool, default False
    For Table formats, append the input data to the existing.
data_columns :  list of columns or True, optional
    List of columns to create as indexed data columns for on-disk
    queries, or True to use all columns. By default only the axes
    of the object are indexed. See `Query via Data Columns
    <https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html#io-hdf5-query-data-columns>`_.
    Applicable only to format='table'.
complevel : {0-9}, optional
    Specifies a compression level for data.
    A value of 0 disables compression.
complib : {'zlib', 'lzo', 'bzip2', 'blosc'}, default 'zlib'
    Specifies the compression library to be used.
    As of v0.20.2 these additional compressors for Blosc are supported
    (default if no compressor specified: 'blosc:blosclz'):
    {'blosc:blosclz', 'blosc:lz4', 'blosc:lz4hc', 'blosc:snappy',
    'blosc:zlib', 'blosc:zstd'}.
    Specifying a compression library which is not available issues
    a ValueError.
fletcher32 : bool, default False
    If applying compression use the fletcher32 checksum.
dropna : bool, default False
    If true, ALL nan rows will not be written to store.
errors : str, default 'strict'
    Specifies how encoding and decoding errors are to be handled.
    See the errors argument for :func:`open` for a full list
    of options.

See Also
--------
cudf.io.hdf.read_hdf : Read from HDF file.
cudf.io.parquet.to_parquet : Write a DataFrame to the binary parquet format.
cudf.io.feather.to_feather : Write out feather-format for DataFrames.
"""
doc_to_hdf = docfmt_partial(docstring=_docstring_to_hdf)

_docstring_read_feather = """
Load an feather object from the file path, returning a DataFrame.

Parameters
----------
path : string
    File path
columns : list, default=None
    If not None, only these columns will be read from the file.

Returns
-------
DataFrame

Examples
--------
>>> import cudf
>>> df = cudf.read_feather(filename)
>>> df
  num1                datetime text
0  123 2018-11-13T12:00:00.000 5451
1  456 2018-11-14T12:35:01.000 5784
2  789 2018-11-15T18:02:59.000 6117

See Also
--------
cudf.io.feather.to_feather
"""
doc_read_feather = docfmt_partial(docstring=_docstring_read_feather)

_docstring_to_feather = """
Write a DataFrame to the feather format.

Parameters
----------
path : str
    File path

See Also
--------
cudf.io.feather.read_feather
"""
doc_to_feather = docfmt_partial(docstring=_docstring_to_feather)

_docstring_to_dlpack = """
Converts a cuDF object into a DLPack tensor.

DLPack is an open-source memory tensor structure:
`dmlc/dlpack <https://github.com/dmlc/dlpack>`_.

This function takes a cuDF object and converts it to a PyCapsule object
which contains a pointer to a DLPack tensor. This function deep copies the
data into the DLPack tensor from the cuDF object.

Parameters
----------
cudf_obj : DataFrame, Series, Index, or Column

Returns
-------
pycapsule_obj : PyCapsule
    Output DLPack tensor pointer which is encapsulated in a PyCapsule
    object.
"""
doc_to_dlpack = docfmt_partial(docstring=_docstring_to_dlpack)

_docstring_read_csv = """
Load a comma-seperated-values (CSV) dataset into a DataFrame

Parameters
----------
filepath_or_buffer : str, path object, or file-like object
    Either a path to a file (a `str`, `pathlib.Path`, or
    `py._path.local.LocalPath`), URL (including http, ftp, and S3 locations),
    or any object with a `read()` method (such as builtin `open()` file handler
    function or `StringIO`).
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
dtype : type, list of types, or dict of column -> type, default None
    Data type(s) for data or columns. If list, types are applied in the same
    order as the column names. If dict, types are mapped to the column names.
    E.g. {{‘a’: np.float64, ‘b’: int32, ‘c’: ‘float’}}
    If `None`, dtypes are inferred from the dataset. Use `str` to preserve data
    and not infer or interpret to dtype.
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
compression : {{'infer', 'gzip', 'zip', None}}, default 'infer'
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
parse_dates : list of int or names, default None
    If list of columns, then attempt to parse each entry as a date.
    Columns may not always be recognized as dates, for instance due to
    unusual or non-standard formats. To guarantee a date and increase parsing
    speed, explicitly specify `dtype='date'` for the desired columns.
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
index_col : int, string or False, default None
    Column to use as the row labels of the DataFrame. Passing `index_col=False`
    explicitly disables index column inference and discards the last column.

Returns
-------
GPU ``DataFrame`` object.

Notes
-----
{remote_data_sources}

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

See Also
--------
cudf.io.csv.to_csv
""".format(
    remote_data_sources=_docstring_remote_sources
)
doc_read_csv = docfmt_partial(docstring=_docstring_read_csv)

_docstring_to_csv = """
Write a dataframe to csv file format.

Parameters
----------
df : DataFrame
    DataFrame object to be written to csv
path : str, default None
    Path of file where DataFrame will be written
sep : char, default ','
    Delimiter to be used.
na_rep : str, default ''
    String to use for null entries
columns : list of str, optional
    Columns to write
header : bool, default True
    Write out the column names
index : bool, default True
    Write out the index as a column
line_terminator : char, default '\\n'
chunksize : int or None, default None
    Rows to write at a time

Notes
-----
- Follows the standard of Pandas csv.QUOTE_NONNUMERIC for all output.
- If `to_csv` leads to memory errors consider setting the `chunksize` argument.

Examples
--------

Write a dataframe to csv.

>>> import cudf
>>> filename = 'foo.csv'
>>> df = cudf.DataFrame({'x': [0, 1, 2, 3],
                         'y': [1.0, 3.3, 2.2, 4.4],
                         'z': ['a', 'b', 'c', 'd']})
>>> df = df.set_index([3, 2, 1, 0])
>>> df.to_csv(filename)

See Also
--------
cudf.io.csv.read_csv
"""
doc_to_csv = docfmt_partial(docstring=_docstring_to_csv)


def is_url(url):
    """Check if a string is a valid URL to a network location.

    Parameters
    ----------
    url : str
        String containing a possible URL

    Returns
    -------
    bool : bool
        If `url` has a valid protocol return True otherwise False.
    """
    # Do not include the empty ('') scheme in the check
    schemes = urllib.parse.uses_netloc[1:]
    try:
        return urllib.parse.urlparse(url).scheme in schemes
    except Exception:
        return False


def is_file_like(obj):
    """Check if the object is a file-like object, per PANDAS' definition.
    An object is considered file-like if it has an iterator AND has a either or
    both `read()` / `write()` methods as attributes.

    Parameters
    ----------
    obj : object
        Object to check for file-like properties

    Returns
    -------
    is_file_like : bool
        If `obj` is file-like returns True otherwise False
    """
    if not (hasattr(obj, "read") or hasattr(obj, "write")):
        return False
    elif not hasattr(obj, "__iter__"):
        return False
    else:
        return True


def _is_local_filesystem(fs):
    local_filesystem_protocols = ["file"]

    if hasattr(fs, "protocol"):
        if fs.protocol in local_filesystem_protocols:
            return True
    else:
        return False


def get_filepath_or_buffer(
    path_or_data, compression, iotypes=(BytesIO), **kwargs
):
    """Return either a filepath string to data, or a memory buffer of data.
    If filepath, then the source filepath is expanded to user's environment.
    If buffer, then data is returned in-memory as bytes or a ByteIO object.

    Parameters
    ----------
    path_or_data : str, file-like object, bytes, ByteIO
        Path to data or the data itself.
    compression : str
        Type of compression algorithm for the content
    iotypes : (), default (BytesIO)
        Object type to exclude from file-like check

    Returns
    -------
    filepath_or_buffer : str, bytes, BytesIO
        Filepath string or in-memory buffer of data
    compression : str
        Type of compression algorithm for the content
    """
    if isinstance(path_or_data, str):
        storage_options = kwargs.get("storage_options")
        # fsspec does not expanduser so handle here
        path_or_data = os.path.expanduser(path_or_data)
        fs, _, paths = fsspec.get_fs_token_paths(
            path_or_data, mode="rb", storage_options=storage_options
        )
        if len(paths) == 0:
            raise IOError(f"{path_or_data} could not be resolved to any files")
        elif len(paths) > 1:
            warnings.warn(
                f"`path_or_data` resolved to more than 1 file. "
                f"Only the first file {paths[0]} will be read.",
                UserWarning,
            )

        if _is_local_filesystem(fs):
            # Doing this as `read_json` accepts a json string
            # path_or_data need not be a filepath like string
            if os.path.exists(paths[0]):
                path_or_data = paths[0]
        else:
            with fs.open(paths[0]) as f:
                path_or_data = BytesIO(f.read())

    elif not isinstance(path_or_data, iotypes) and is_file_like(path_or_data):
        if isinstance(path_or_data, TextIOWrapper):
            path_or_data = path_or_data.buffer
        path_or_data = BytesIO(path_or_data.read())

    return path_or_data, compression
