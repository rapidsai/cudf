# Copyright (c) 2019, NVIDIA CORPORATION.

import cudf

import pandas as pd
import warnings


def read_json(path_or_buf, *args, **kwargs):
    """
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
    .to_json
    """

    warnings.warn("Using CPU via Pandas to read JSON dataset, this may "
                  "be GPU accelerated in the future")

    pd_value = pd.read_json(path_or_buf, *args, **kwargs)
    return cudf.from_pandas(pd_value)


def to_json(cudf_val, path_or_buf=None, *args, **kwargs):
    """
    Convert the cuDF object to a JSON string.
    Note nulls and NaNs will be converted to null and datetime objects
    will be converted to UNIX timestamps.
    Parameters
    ----------
    path_or_buf : string or file handle, optional
        File path or object. If not specified, the result is returned as
        a string.
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
    .read_json
    """

    warnings.warn("Using CPU via Pandas to write JSON dataset, this may "
                  "be GPU accelerated in the future")

    pd_value = cudf_val.to_pandas()
    pd.io.json.to_json(
        path_or_buf,
        pd_value,
        *args,
        **kwargs
    )
