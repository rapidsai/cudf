# Copyright (c) 2019-2022, NVIDIA CORPORATION.

import datetime
import os
import urllib
import warnings
from io import BufferedWriter, BytesIO, IOBase, TextIOWrapper
from threading import Thread

import fsspec
import fsspec.implementations.local
import numpy as np
import pandas as pd
from fsspec.core import get_fs_token_paths
from pyarrow import PythonFile as ArrowPythonFile
from pyarrow.fs import FSSpecHandler, PyFileSystem
from pyarrow.lib import NativeFile

from cudf.utils.docutils import docfmt_partial

try:
    import fsspec.parquet as fsspec_parquet

except ImportError:
    fsspec_parquet = None


_docstring_remote_sources = """
- cuDF supports local and remote data stores. See configuration details for
  available sources
  `here <https://docs.dask.org/en/latest/remote-data-services.html>`__.
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
engine : ['cudf'], default 'cudf'
    Parser engine to use.
columns : list, default None
    If not None, only these columns will be read.
skiprows : int, default None
    If not None, the number of rows to skip from the start of the file.
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
>>> import pandavro
>>> import pandas as pd
>>> import cudf
>>> pandas_df = pd.DataFrame()
>>> pandas_df['numbers'] = [10, 20, 30]
>>> pandas_df['text'] = ["hello", "rapids", "ai"]
>>> pandas_df
   numbers    text
0       10   hello
1       20  rapids
2       30      ai
>>> pandavro.to_avro("data.avro", pandas_df)
>>> cudf.read_avro("data.avro")
   numbers    text
0       10   hello
1       20  rapids
2       30      ai
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
cudf.read_parquet
"""
doc_read_parquet_metadata = docfmt_partial(
    docstring=_docstring_read_parquet_metadata
)

_docstring_read_parquet = """
Load a Parquet dataset into a DataFrame

Parameters
----------
filepath_or_buffer : str, path object, bytes, file-like object, or a list
    of such objects.
    Contains one or more of the following: either a path to a file (a `str`,
    `pathlib.Path`, or `py._path.local.LocalPath`), URL (including http, ftp,
    and S3 locations), Python bytes of raw binary data, or any object with a
    `read()` method (such as builtin `open()` file handler function or
    `BytesIO`).
engine : {{ 'cudf', 'pyarrow' }}, default 'cudf'
    Parser engine to use.
columns : list, default None
    If not None, only these columns will be read.
filters : list of tuple, list of lists of tuples default None
    If not None, specifies a filter predicate used to filter out row groups
    using statistics stored for each row group as Parquet metadata. Row groups
    that do not match the given filter predicate are not read. The
    predicate is expressed in disjunctive normal form (DNF) like
    `[[('x', '=', 0), ...], ...]`. DNF allows arbitrary boolean logical
    combinations of single column predicates. The innermost tuples each
    describe a single column predicate. The list of inner predicates is
    interpreted as a conjunction (AND), forming a more selective and
    multiple column predicate. Finally, the most outer list combines
    these filters as a disjunction (OR). Predicates may also be passed
    as a list of tuples. This form is interpreted as a single conjunction.
    To express OR in predicates, one must use the (preferred) notation of
    list of lists of tuples.
row_groups : int, or list, or a list of lists default None
    If not None, specifies, for each input file, which row groups to read.
    If reading multiple inputs, a list of lists should be passed, one list
    for each input.
skiprows : int, default None
    If not None, the number of rows to skip from the start of the file.
num_rows : int, default None
    If not None, the total number of rows to read.
strings_to_categorical : boolean, default False
    If True, return string columns as GDF_CATEGORY dtype; if False, return a
    as GDF_STRING dtype.
categorical_partitions : boolean, default True
    Whether directory-partitioned columns should be interpreted as categorical
    or raw dtypes.
use_pandas_metadata : boolean, default True
    If True and dataset has custom PANDAS schema metadata, ensure that index
    columns are also loaded.
use_python_file_object : boolean, default True
    If True, Arrow-backed PythonFile objects will be used in place of fsspec
    AbstractBufferedFile objects at IO time. Setting this argument to `False`
    will require the entire file to be copied to host memory, and is highly
    discouraged.
open_file_options : dict, optional
    Dictionary of key-value pairs to pass to the function used to open remote
    files. By default, this will be `fsspec.parquet.open_parquet_file`. To
    deactivate optimized precaching, set the "method" to `None` under the
    "precache_options" key. Note that the `open_file_func` key can also be
    used to specify a custom file-open function.

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
cudf.DataFrame.to_parquet
cudf.read_orc
""".format(
    remote_data_sources=_docstring_remote_sources
)
doc_read_parquet = docfmt_partial(docstring=_docstring_read_parquet)

_docstring_to_parquet = """
Write a DataFrame to the parquet format.

Parameters
----------
path : str or list of str
    File path or Root Directory path. Will be used as Root Directory path
    while writing a partitioned dataset. Use list of str with partition_offsets
    to write parts of the dataframe to different files.
compression : {'snappy', None}, default 'snappy'
    Name of the compression to use. Use ``None`` for no compression.
index : bool, default None
    If ``True``, include the dataframe's index(es) in the file output. If
    ``False``, they will not be written to the file. If ``None``, the
    engine's default behavior will be used. However, instead of being saved
    as values, the ``RangeIndex`` will be stored as a range in the metadata
    so it doesn’t require much space and is faster. Other indexes will
    be included as columns in the file output.
partition_cols : list, optional, default None
    Column names by which to partition the dataset
    Columns are partitioned in the order they are given
partition_file_name : str, optional, default None
    File name to use for partitioned datasets. Different partitions
    will be written to different directories, but all files will
    have this name.  If nothing is specified, a random uuid4 hex string
    will be used for each file.
partition_offsets : list, optional, default None
    Offsets to partition the dataframe by. Should be used when path is list
    of str. Should be a list of integers of size ``len(path) + 1``
statistics : {'ROWGROUP', 'PAGE', 'NONE'}, default 'ROWGROUP'
    Level at which column statistics should be included in file.
metadata_file_path : str, optional, default None
    If specified, this function will return a binary blob containing the footer
    metadata of the written parquet file. The returned blob will have the
    ``chunk.file_path`` field set to the ``metadata_file_path`` for each chunk.
    When using with ``partition_offsets``, should be same size as ``len(path)``
int96_timestamps : bool, default False
    If ``True``, write timestamps in int96 format. This will convert
    timestamps from timestamp[ns], timestamp[ms], timestamp[s], and
    timestamp[us] to the int96 format, which is the number of Julian
    days and the number of nanoseconds since midnight. If ``False``,
    timestamps will not be altered.
row_group_size_bytes: integer or None, default None
    Maximum size of each stripe of the output.
    If None, 134217728 (128MB) will be used.
row_group_size_rows: integer or None, default None
    Maximum number of rows of each stripe of the output.
    If None, 1000000 will be used.
**kwargs
    To request metadata binary blob when using with ``partition_cols``, Pass
    ``return_metadata=True`` instead of specifying ``metadata_file_path``


See Also
--------
cudf.read_parquet
"""
doc_to_parquet = docfmt_partial(docstring=_docstring_to_parquet)

_docstring_merge_parquet_filemetadata = """
Merge multiple parquet metadata blobs

Parameters
----------
metadata_list : list
    List of buffers returned by to_parquet

Returns
-------
Combined parquet metadata blob

See Also
--------
cudf.DataFrame.to_parquet
"""
doc_merge_parquet_filemetadata = docfmt_partial(
    docstring=_docstring_merge_parquet_filemetadata
)


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

Notes
-----
Support for reading files with struct columns is currently experimental,
the output may not be as reliable as reading for other datatypes.
{remote_data_sources}

Examples
--------
>>> import cudf
>>> num_rows, stripes, names = cudf.io.read_orc_metadata(filename)
>>> df = [cudf.read_orc(fname, stripes=i) for i in range(stripes)]
>>> df = cudf.concat(df)
>>> df
  num1                datetime text
0  123 2018-11-13T12:00:00.000 5451
1  456 2018-11-14T12:35:01.000 5784
2  789 2018-11-15T18:02:59.000 6117

See Also
--------
cudf.read_orc
"""
doc_read_orc_metadata = docfmt_partial(docstring=_docstring_read_orc_metadata)


_docstring_read_orc_statistics = """
Read an ORC file's file-level and stripe-level statistics

Parameters
----------
filepath_or_buffer : str, path object, bytes, or file-like object
    Either a path to a file (a `str`, `pathlib.Path`, or
    `py._path.local.LocalPath`), URL (including http, ftp, and S3 locations),
    Python bytes of raw binary data, or any object with a `read()` method
    (such as builtin `open()` file handler function or `BytesIO`).
columns : list, default None
    If not None, statistics for only these columns will be read from the file.


Returns
-------
Statistics for each column of given file
Statistics for each column for each stripe of given file

See Also
--------
cudf.read_orc
"""
doc_read_orc_statistics = docfmt_partial(
    docstring=_docstring_read_orc_statistics
)

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
filters : list of tuple, list of lists of tuples default None
    If not None, specifies a filter predicate used to filter out row groups
    using statistics stored for each row group as Parquet metadata. Row groups
    that do not match the given filter predicate are not read. The
    predicate is expressed in disjunctive normal form (DNF) like
    `[[('x', '=', 0), ...], ...]`. DNF allows arbitrary boolean logical
    combinations of single column predicates. The innermost tuples each
    describe a single column predicate. The list of inner predicates is
    interpreted as a conjunction (AND), forming a more selective and
    multiple column predicate. Finally, the outermost list combines
    these filters as a disjunction (OR). Predicates may also be passed
    as a list of tuples. This form is interpreted as a single conjunction.
    To express OR in predicates, one must use the (preferred) notation of
    list of lists of tuples.
stripes: list, default None
    If not None, only these stripe will be read from the file. Stripes are
    concatenated with index ignored.
skiprows : int, default None
    If not None, the number of rows to skip from the start of the file.
num_rows : int, default None
    If not None, the total number of rows to read.
use_index : bool, default True
    If True, use row index if available for faster seeking.
use_python_file_object : boolean, default True
    If True, Arrow-backed PythonFile objects will be used in place of fsspec
    AbstractBufferedFile objects at IO time. This option is likely to improve
    performance when making small reads from larger ORC files.
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
cudf.DataFrame.to_orc
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
enable_statistics: boolean, default True
    Enable writing column statistics.
stripe_size_bytes: integer or None, default None
    Maximum size of each stripe of the output.
    If None, 67108864 (64MB) will be used.
stripe_size_rows: integer or None, default None
    Maximum number of rows of each stripe of the output.
    If None, 1000000 will be used.
row_index_stride: integer or None, default None
    Row index stride (maximum number of rows in each row group).
    If None, 10000 will be used.


Notes
-----
Support for writing tables with struct columns is currently experimental,
the output may not be as reliable as writing for other datatypes.

See Also
--------
cudf.read_orc
"""
doc_to_orc = docfmt_partial(docstring=_docstring_to_orc)

_docstring_read_json = """
Load a JSON dataset into a DataFrame

Parameters
----------
path_or_buf : list, str, path object, or file-like object
    Either JSON data in a `str`, path to a file (a `str`, `pathlib.Path`, or
    `py._path.local.LocalPath`), URL (including http, ftp, and S3 locations),
    or any object with a `read()` method (such as builtin `open()` file handler
    function or `StringIO`). Multiple inputs may be provided as a list. If a
    list is specified each list entry may be of a different input type as long
    as each input is of a valid type and all input JSON schema(s) match.
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
    The default behavior is to try and detect the correct precision, but if
    this is not desired then pass one of 's', 'ms', 'us' or 'ns' to force
    parsing only seconds, milliseconds, microseconds or nanoseconds.
encoding : str, default is 'utf-8'
    The encoding to use to decode py3 bytes.
    With cudf engine, only utf-8 is supported.
lines : boolean, default False
    Read the file as a json object per line.
chunksize : integer, default None
    Return JsonReader object for iteration (pandas engine only).
    See the `line-delimited json docs
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
cudf.DataFrame.to_json
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
    the object to convert and return a serializable object.
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
cudf.read_json
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
cudf.DataFrame.to_hdf : Write a HDF file from a DataFrame.
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
cudf.read_hdf : Read from HDF file.
cudf.DataFrame.to_parquet : Write a DataFrame to the binary parquet format.
cudf.DataFrame.to_feather : Write out feather-format for DataFrames.
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
cudf.DataFrame.to_feather
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
cudf.read_feather
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
dtype : type, str, list of types, or dict of column -> type, default None
    Data type(s) for data or columns. If `dtype` is a type/str, all columns
    are mapped to the particular type passed. If list, types are applied in
    the same order as the column names. If dict, types are mapped to the
    column names.
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
na_values : scalar, str, or list-like, optional
    Additional strings to recognize as nulls.
    By default the following values are interpreted as
    nulls: '', '#N/A', '#N/A N/A', '#NA', '-1.#IND',
    '-1.#QNAN', '-NaN', '-nan', '1.#IND', '1.#QNAN',
    '<NA>', 'N/A', 'NA', 'NULL', 'NaN', 'n/a', 'nan',
    'null'.
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
use_python_file_object : boolean, default True
    If True, Arrow-backed PythonFile objects will be used in place of fsspec
    AbstractBufferedFile objects at IO time. This option is likely to improve
    performance when making small reads from larger CSV files.

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
cudf.DataFrame.to_csv
""".format(
    remote_data_sources=_docstring_remote_sources
)
doc_read_csv = docfmt_partial(docstring=_docstring_read_csv)

_to_csv_example = """

Write a dataframe to csv.

>>> import cudf
>>> filename = 'foo.csv'
>>> df = cudf.DataFrame({'x': [0, 1, 2, 3],
...                      'y': [1.0, 3.3, 2.2, 4.4],
...                      'z': ['a', 'b', 'c', 'd']})
>>> df = df.set_index(cudf.Series([3, 2, 1, 0]))
>>> df.to_csv(filename)

"""
_docstring_to_csv = """

Write a dataframe to csv file format.

Parameters
----------
{df_param}
path_or_buf : str or file handle, default None
    File path or object, if None is provided
    the result is returned as a string.
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
encoding: str, default 'utf-8'
    A string representing the encoding to use in the output file
    Only ‘utf-8’ is currently supported
compression: str, None
    A string representing the compression scheme to use in the the output file
    Compression while writing csv is not supported currently
Returns
-------
None or str
    If `path_or_buf` is None, returns the resulting csv format as a string.
    Otherwise returns None.

Notes
-----
- Follows the standard of Pandas csv.QUOTE_NONNUMERIC for all output.
- If `to_csv` leads to memory errors consider setting the `chunksize` argument.

Examples
--------
{example}

See Also
--------
cudf.read_csv
"""
doc_to_csv = docfmt_partial(
    docstring=_docstring_to_csv.format(
        df_param="""
df : DataFrame
    DataFrame object to be written to csv
""",
        example=_to_csv_example,
    )
)

doc_dataframe_to_csv = docfmt_partial(
    docstring=_docstring_to_csv.format(df_param="", example=_to_csv_example)
)

_docstring_kafka_datasource = """
Configuration object for a Kafka Datasource

Parameters
----------
kafka_configs : dict, key/value pairs of librdkafka configuration values.
    The complete list of valid configurations can be found at
    https://github.com/edenhill/librdkafka/blob/master/CONFIGURATION.md
topic : string, case sensitive name of the Kafka topic that contains the
    source data.
partition : int,
    Zero-based identifier of the Kafka partition that the underlying consumer
    should consume messages from. Valid values are 0 - (N-1)
start_offset : int, Kafka Topic/Partition offset that consumption
    should begin at. Inclusive.
end_offset : int, Kafka Topic/Partition offset that consumption
    should end at. Inclusive.
batch_timeout : int, default 10000
    Maximum number of milliseconds that will be spent trying to
    consume messages between the specified 'start_offset' and 'end_offset'.
delimiter : string, default None, optional delimiter to insert into the
    output between kafka messages, Ex: "\n"

"""
doc_kafka_datasource = docfmt_partial(docstring=_docstring_kafka_datasource)


_docstring_text_datasource = """
Configuration object for a text Datasource

Parameters
----------
filepath_or_buffer : str, path object, or file-like object
    Either a path to a file (a `str`, `pathlib.Path`, or
    `py._path.local.LocalPath`), URL (including http, ftp, and S3 locations),
    or any object with a `read()` method (such as builtin `open()` file handler
    function or `StringIO`).
delimiter : string, default None, The delimiter that should be used
    for splitting text chunks into separate cudf column rows. Currently
    only a single delimiter is supported.

Returns
-------
result : GPU ``Series``

"""
doc_read_text = docfmt_partial(docstring=_docstring_text_datasource)


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
    return isinstance(fs, fsspec.implementations.local.LocalFileSystem)


def ensure_single_filepath_or_buffer(path_or_data, **kwargs):
    """Return False if `path_or_data` resolves to multiple filepaths or
    buffers.
    """
    path_or_data = stringify_pathlike(path_or_data)
    if isinstance(path_or_data, str):
        storage_options = kwargs.get("storage_options")
        path_or_data = os.path.expanduser(path_or_data)
        try:
            fs, _, paths = get_fs_token_paths(
                path_or_data, mode="rb", storage_options=storage_options
            )
        except ValueError as e:
            if str(e).startswith("Protocol not known"):
                return True
            else:
                raise e

        if len(paths) > 1:
            return False
    elif isinstance(path_or_data, (list, tuple)) and len(path_or_data) > 1:
        return False

    return True


def is_directory(path_or_data, **kwargs):
    """Returns True if the provided filepath is a directory"""
    path_or_data = stringify_pathlike(path_or_data)
    if isinstance(path_or_data, str):
        storage_options = kwargs.get("storage_options")
        path_or_data = os.path.expanduser(path_or_data)
        try:
            fs = get_fs_token_paths(
                path_or_data, mode="rb", storage_options=storage_options
            )[0]
        except ValueError as e:
            if str(e).startswith("Protocol not known"):
                return False
            else:
                raise e

        return fs.isdir(path_or_data)

    return False


def _get_filesystem_and_paths(path_or_data, **kwargs):
    # Returns a filesystem object and the filesystem-normalized
    # paths. If `path_or_data` does not correspond to a path or
    # list of paths (or if the protocol is not supported), the
    # return will be `None` for the fs and `[]` for the paths.

    fs = None
    return_paths = path_or_data
    if isinstance(path_or_data, str) or (
        isinstance(path_or_data, list)
        and isinstance(stringify_pathlike(path_or_data[0]), str)
    ):
        # Ensure we are always working with a list
        storage_options = kwargs.get("storage_options")
        if isinstance(path_or_data, list):
            path_or_data = [
                os.path.expanduser(stringify_pathlike(source))
                for source in path_or_data
            ]
        else:
            path_or_data = [path_or_data]

        try:
            fs, _, fs_paths = get_fs_token_paths(
                path_or_data, mode="rb", storage_options=storage_options
            )
            return_paths = fs_paths
        except ValueError as e:
            if str(e).startswith("Protocol not known"):
                return None, []
            else:
                raise e

    return fs, return_paths


def _set_context(obj, stack):
    # Helper function to place open file on context stack
    if stack is None:
        return obj
    return stack.enter_context(obj)


def _open_remote_files(
    paths,
    fs,
    context_stack=None,
    open_file_func=None,
    precache_options=None,
    **kwargs,
):
    """Return a list of open file-like objects given
    a list of remote file paths.

    Parameters
    ----------
    paths : list(str)
        List of file-path strings.
    fs : fsspec.AbstractFileSystem
        Fsspec file-system object.
    context_stack : contextlib.ExitStack, Optional
        Context manager to use for open files.
    open_file_func : Callable, Optional
        Call-back function to use for opening. If this argument
        is specified, all other arguments will be ignored.
    precache_options : dict, optional
        Dictionary of key-word arguments to pass to use for
        precaching. Unless the input contains ``{"method": None}``,
        ``fsspec.parquet.open_parquet_file`` will be used for remote
        storage.
    **kwargs :
        Key-word arguments to be passed to format-specific
        open functions.
    """

    # Just use call-back function if one was specified
    if open_file_func is not None:
        return [
            _set_context(open_file_func(path, **kwargs), context_stack)
            for path in paths
        ]

    # Check if the "precache" option is supported.
    # In the future, fsspec should do this check for us
    precache_options = (precache_options or {}).copy()
    precache = precache_options.pop("method", None)
    if precache not in ("parquet", None):
        raise ValueError(f"{precache} not a supported `precache` option.")

    # Check that "parts" caching (used for all format-aware file handling)
    # is supported by the installed fsspec/s3fs version
    if precache == "parquet" and not fsspec_parquet:
        warnings.warn(
            f"This version of fsspec ({fsspec.__version__}) does "
            f"not support parquet-optimized precaching. Please upgrade "
            f"to the latest fsspec version for better performance."
        )
        precache = None

    if precache == "parquet":
        # Use fsspec.parquet module.
        # TODO: Use `cat_ranges` to collect "known"
        # parts for all files at once.
        row_groups = precache_options.pop("row_groups", None) or (
            [None] * len(paths)
        )
        return [
            ArrowPythonFile(
                _set_context(
                    fsspec_parquet.open_parquet_file(
                        path,
                        fs=fs,
                        row_groups=rgs,
                        **precache_options,
                        **kwargs,
                    ),
                    context_stack,
                )
            )
            for path, rgs in zip(paths, row_groups)
        ]

    # Default open - Use pyarrow filesystem API
    pa_fs = PyFileSystem(FSSpecHandler(fs))
    return [
        _set_context(pa_fs.open_input_file(fpath), context_stack)
        for fpath in paths
    ]


def get_reader_filepath_or_buffer(
    path_or_data,
    compression,
    mode="rb",
    fs=None,
    iotypes=(BytesIO, NativeFile),
    byte_ranges=None,
    use_python_file_object=False,
    open_file_options=None,
    allow_raw_text_input=False,
    **kwargs,
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
    mode : str
        Mode in which file is opened
    iotypes : (), default (BytesIO)
        Object type to exclude from file-like check
    byte_ranges : list, optional
        List of known byte ranges that will be read from path_or_data
    use_python_file_object : boolean, default False
        If True, Arrow-backed PythonFile objects will be used in place
        of fsspec AbstractBufferedFile objects.
    open_file_options : dict, optional
        Optional dictionary of key-word arguments to pass to
        `_open_remote_files` (used for remote storage only).
    allow_raw_text_input : boolean, default False
        If True, this indicates the input `path_or_data` could be a raw text
        input and will not check for its existence in the filesystem. If False,
        the input must be a path and an error will be raised if it does not
        exist.

    Returns
    -------
    filepath_or_buffer : str, bytes, BytesIO, list
        Filepath string or in-memory buffer of data or a
        list of Filepath strings or in-memory buffers of data.
    compression : str
        Type of compression algorithm for the content
    """
    path_or_data = stringify_pathlike(path_or_data)

    if isinstance(path_or_data, str):

        # Get a filesystem object if one isn't already available
        paths = [path_or_data]
        if fs is None:
            fs, paths = _get_filesystem_and_paths(path_or_data, **kwargs)
            if fs is None:
                return path_or_data, compression

        if _is_local_filesystem(fs):
            # Doing this as `read_json` accepts a json string
            # path_or_data need not be a filepath like string
            if len(paths):
                if fs.exists(paths[0]):
                    path_or_data = paths if len(paths) > 1 else paths[0]
                elif not allow_raw_text_input:
                    raise FileNotFoundError(
                        f"{path_or_data} could not be resolved to any files"
                    )

        else:
            if len(paths) == 0:
                raise FileNotFoundError(
                    f"{path_or_data} could not be resolved to any files"
                )
            if use_python_file_object:
                path_or_data = _open_remote_files(
                    paths,
                    fs,
                    **(open_file_options or {}),
                )
            else:
                path_or_data = [
                    BytesIO(
                        _fsspec_data_transfer(
                            fpath,
                            fs=fs,
                            mode=mode,
                            **kwargs,
                        )
                    )
                    for fpath in paths
                ]
            if len(path_or_data) == 1:
                path_or_data = path_or_data[0]

    elif not isinstance(path_or_data, iotypes) and is_file_like(path_or_data):
        if isinstance(path_or_data, TextIOWrapper):
            path_or_data = path_or_data.buffer
        if use_python_file_object:
            path_or_data = ArrowPythonFile(path_or_data)
        else:
            path_or_data = BytesIO(
                _fsspec_data_transfer(path_or_data, mode=mode, **kwargs)
            )

    return path_or_data, compression


def get_writer_filepath_or_buffer(path_or_data, mode, **kwargs):
    """
    Return either a filepath string to data,
    or a open file object to the output filesystem

    Parameters
    ----------
    path_or_data : str, file-like object, bytes, ByteIO
        Path to data or the data itself.
    mode : str
        Mode in which file is opened

    Returns
    -------
    filepath_or_buffer : str,
        Filepath string or buffer of data
    """
    if isinstance(path_or_data, str):
        storage_options = kwargs.get("storage_options", {})
        path_or_data = os.path.expanduser(path_or_data)
        fs = get_fs_token_paths(
            path_or_data, mode=mode or "w", storage_options=storage_options
        )[0]

        if not _is_local_filesystem(fs):
            filepath_or_buffer = fsspec.open(
                path_or_data, mode=mode or "w", **(storage_options)
            )
            return filepath_or_buffer

    return path_or_data


def get_IOBase_writer(file_obj):
    """
    Parameters
    ----------
    file_obj : file-like object
        Open file object for writing to any filesystem

    Returns
    -------
    iobase_file_obj : file-like object
        Open file object inheriting from io.IOBase
    """
    if not isinstance(file_obj, IOBase):
        if "b" in file_obj.mode:
            iobase_file_obj = BufferedWriter(file_obj)
        else:
            iobase_file_obj = TextIOWrapper(file_obj)
        return iobase_file_obj

    return file_obj


def is_fsspec_open_file(file_obj):
    if isinstance(file_obj, fsspec.core.OpenFile):
        return True
    return False


def stringify_pathlike(pathlike):
    """
    Convert any object that implements the fspath protocol
    to a string. Leaves other objects unchanged
    Parameters
    ----------
    pathlike
        Pathlike object that implements the fspath protocol

    Returns
    -------
    maybe_pathlike_str
        String version of the object if possible
    """
    maybe_pathlike_str = (
        pathlike.__fspath__() if hasattr(pathlike, "__fspath__") else pathlike
    )

    return maybe_pathlike_str


def buffer_write_lines(buf, lines):
    """
    Appends lines to a buffer.
    Parameters
    ----------
    buf
        The buffer to write to
    lines
        The lines to append.
    """
    if any(isinstance(x, str) for x in lines):
        lines = [str(x) for x in lines]
    buf.write("\n".join(lines))


def _apply_filter_bool_eq(val, col_stats):
    if "true_count" in col_stats and "false_count" in col_stats:
        if val is True:
            if (col_stats["true_count"] == 0) or (
                col_stats["false_count"] == col_stats["number_of_values"]
            ):
                return False
        elif val is False:
            if (col_stats["false_count"] == 0) or (
                col_stats["true_count"] == col_stats["number_of_values"]
            ):
                return False
    return True


def _apply_filter_not_eq(val, col_stats):
    return ("minimum" in col_stats and val < col_stats["minimum"]) or (
        "maximum" in col_stats and val > col_stats["maximum"]
    )


def _apply_predicate(op, val, col_stats):
    # Sanitize operator
    if op not in {"=", "==", "!=", "<", "<=", ">", ">=", "in", "not in"}:
        raise ValueError(f"'{op}' is not a valid operator in predicates.")

    col_min = col_stats.get("minimum", None)
    col_max = col_stats.get("maximum", None)
    col_sum = col_stats.get("sum", None)

    # Apply operator
    if op == "=" or op == "==":
        if _apply_filter_not_eq(val, col_stats):
            return False
        # TODO: Replace pd.isnull with
        # cudf.isnull once it is implemented
        if pd.isnull(val) and not col_stats["has_null"]:
            return False
        if not _apply_filter_bool_eq(val, col_stats):
            return False
    elif op == "!=":
        if (
            col_min is not None
            and col_max is not None
            and val == col_min
            and val == col_max
        ):
            return False
        if _apply_filter_bool_eq(val, col_stats):
            return False
    elif col_min is not None and (
        (op == "<" and val <= col_min) or (op == "<=" and val < col_min)
    ):
        return False
    elif col_max is not None and (
        (op == ">" and val >= col_max) or (op == ">=" and val > col_max)
    ):
        return False
    elif (
        col_sum is not None
        and op == ">"
        and (
            (col_min is not None and col_min >= 0 and col_sum <= val)
            or (col_max is not None and col_max <= 0 and col_sum >= val)
        )
    ):
        return False
    elif (
        col_sum is not None
        and op == ">="
        and (
            (col_min is not None and col_min >= 0 and col_sum < val)
            or (col_max is not None and col_max <= 0 and col_sum > val)
        )
    ):
        return False
    elif op == "in":
        if (col_max is not None and col_max < min(val)) or (
            col_min is not None and col_min > max(val)
        ):
            return False
        if all(_apply_filter_not_eq(elem, col_stats) for elem in val):
            return False
    elif op == "not in" and col_min is not None and col_max is not None:
        if any(elem == col_min == col_max for elem in val):
            return False
        col_range = None
        if isinstance(col_min, int):
            col_range = range(col_min, col_max)
        elif isinstance(col_min, datetime.datetime):
            col_range = pd.date_range(col_min, col_max)
        if col_range and all(elem in val for elem in col_range):
            return False
    return True


def _apply_filters(filters, stats):
    for conjunction in filters:
        if all(
            _apply_predicate(op, val, stats[col])
            for col, op, val in conjunction
        ):
            return True
    return False


def _prepare_filters(filters):
    # Coerce filters into list of lists of tuples
    if isinstance(filters[0][0], str):
        filters = [filters]

    return filters


def _ensure_filesystem(passed_filesystem, path, **kwargs):
    if passed_filesystem is None:
        return get_fs_token_paths(
            path[0] if isinstance(path, list) else path,
            storage_options=kwargs.get("storage_options", {}),
        )[0]
    return passed_filesystem


#
# Fsspec Data-transfer Optimization Code
#


def _fsspec_data_transfer(
    path_or_fob,
    fs=None,
    file_size=None,
    bytes_per_thread=256_000_000,
    max_gap=64_000,
    mode="rb",
    **kwargs,
):

    # Require `fs` if `path_or_fob` is not file-like
    file_like = is_file_like(path_or_fob)
    if fs is None and not file_like:
        raise ValueError(
            "fs must be defined if `path_or_fob` is not file-like"
        )

    # Calculate total file size
    if file_like:
        file_size = path_or_fob.size
    file_size = file_size or fs.size(path_or_fob)

    # Check if a direct read makes the most sense
    if bytes_per_thread >= file_size:
        if file_like:
            return path_or_fob.read()
        else:
            return fs.open(path_or_fob, mode=mode, cache_type="all").read()

    # Threaded read into "local" buffer
    buf = np.zeros(file_size, dtype="b")

    byte_ranges = [
        (b, min(bytes_per_thread, file_size - b))
        for b in range(0, file_size, bytes_per_thread)
    ]
    _read_byte_ranges(
        path_or_fob,
        byte_ranges,
        buf,
        fs=fs,
        **kwargs,
    )

    return buf.tobytes()


def _merge_ranges(byte_ranges, max_block=256_000_000, max_gap=64_000):
    # Simple utility to merge small/adjacent byte ranges
    new_ranges = []
    if not byte_ranges:
        # Early return
        return new_ranges

    offset, size = byte_ranges[0]
    for (new_offset, new_size) in byte_ranges[1:]:
        gap = new_offset - (offset + size)
        if gap > max_gap or (size + new_size + gap) > max_block:
            # Gap is too large or total read is too large
            new_ranges.append((offset, size))
            offset = new_offset
            size = new_size
            continue
        size += new_size + gap
    new_ranges.append((offset, size))
    return new_ranges


def _assign_block(fs, path_or_fob, local_buffer, offset, nbytes):
    if fs is None:
        # We have an open fsspec file object
        path_or_fob.seek(offset)
        local_buffer[offset : offset + nbytes] = np.frombuffer(
            path_or_fob.read(nbytes),
            dtype="b",
        )
    else:
        # We have an fsspec filesystem and a path
        with fs.open(path_or_fob, mode="rb", cache_type="none") as fob:
            fob.seek(offset)
            local_buffer[offset : offset + nbytes] = np.frombuffer(
                fob.read(nbytes),
                dtype="b",
            )


def _read_byte_ranges(
    path_or_fob,
    ranges,
    local_buffer,
    fs=None,
    **kwargs,
):
    # Simple utility to copy remote byte ranges
    # into a local buffer for IO in libcudf
    workers = []
    for (offset, nbytes) in ranges:
        if len(ranges) > 1:
            workers.append(
                Thread(
                    target=_assign_block,
                    args=(fs, path_or_fob, local_buffer, offset, nbytes),
                )
            )
            workers[-1].start()
        else:
            _assign_block(fs, path_or_fob, local_buffer, offset, nbytes)

    for worker in workers:
        worker.join()
