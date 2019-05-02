# Copyright (c) 2019, NVIDIA CORPORATION.

from cudf.utils.docutils import docfmt_partial

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
>>> num_rows, num_row_groups, names = cudf.read_parquet_metadata(filename)
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
    docstring=_docstring_read_parquet_metadata)

_docstring_read_parquet = """
Read a Parquet file into DataFrame

Parameters
----------
path : string or path object
    Path of file to be read
engine : { 'cudf', 'pyarrow' }, default 'cudf'
    Parser engine to use.
columns : list, default None
    If not None, only these columns will be read.
row_group : int, default None
    If not None, only the row group with the specified index will be read.
skip_rows : int, default None
    If not None, the nunber of rows to skip from the start of the file.
num_rows : int, default None
    If not None, the total number of rows to read.

Returns
-------
DataFrame

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
"""
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

_docstring_read_orc = """
Load an ORC object from the file path, returning a DataFrame.

Parameters
----------
path : string
    File path
engine : { 'cudf', 'pyarrow' }, default 'pyarrow'
    Parser engine to use.
columns : list, default None
    If not None, only these columns will be read from the file.
skip_rows : int, default None
    If not None, the number of rows to skip from the start of the file.
num_rows : int, default None
    If not None, the total number of rows to read.
kwargs are passed to the engine

Returns
-------
DataFrame

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
"""
doc_read_orc = docfmt_partial(docstring=_docstring_read_orc)

_docstring_read_json = """
Convert a JSON string to a cuDF object.

Parameters
----------
path_or_buf : a valid JSON string or file-like, default: None
    The string could be a URL. Valid URL schemes include http, ftp, s3,
    gcs, and file. For file URLs, a host is expected. For instance, a local
    file could be ``file://localhost/path/to/table.json``
orient : string,
    Indication of expected JSON string format.
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
       'table' as an allowed value for the ``orient`` argument
typ : type of object to recover (series or frame), default 'frame'
dtype : boolean or dict, default True
    If True, infer dtypes, if a dict of column to dtype, then use those,
    if False, then don't infer dtypes at all, applies only to the data.
convert_axes : boolean, default True
    Try to convert the axes to the proper dtypes.
convert_dates : boolean, default True
    List of columns to parse for dates; If True, then try to parse
    datelike columns default is True; a column label is datelike if
    * it ends with ``'_at'``,
    * it ends with ``'_time'``,
    * it begins with ``'timestamp'``,
    * it is ``'modified'``, or
    * it is ``'date'``
keep_default_dates : boolean, default True
    If parsing dates, then parse the default datelike columns
numpy : boolean, default False
    Direct decoding to numpy arrays. Supports numeric data only, but
    non-numeric column and index labels are supported. Note also that the
    JSON ordering MUST be the same for each term if numpy=True.
precise_float : boolean, default False
    Set to enable usage of higher precision (strtod) function when
    decoding string to double values. Default (False) is to use fast but
    less precise builtin functionality
date_unit : string, default None
    The timestamp unit to detect if converting dates. The default behaviour
    is to try and detect the correct precision, but if this is not desired
    then pass one of 's', 'ms', 'us' or 'ns' to force parsing only seconds,
    milliseconds, microseconds or nanoseconds respectively.
encoding : str, default is 'utf-8'
    The encoding to use to decode py3 bytes.
lines : boolean, default False
    Read the file as a json object per line.
chunksize : integer, default None
    Return JsonReader object for iteration.
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
cudf.io.feather..to_feather : Write out feather-format for DataFrames.
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
