# Copyright (c) 2019, NVIDIA CORPORATION.

import cudf

import pandas as pd
import warnings


def read_hdf(path_or_buf, key=None, mode='r', **kwargs):
    """
    Read from the store, close it if we opened it.

    Retrieve pandas object stored in file, optionally based on where
    criteria

    Parameters
    ----------
    path_or_buf : string, buffer or path object
        Path to the file to open, or an open :class:`pandas.HDFStore` object.
        Supports any object implementing the ``__fspath__`` protocol.
        This includes :class:`pathlib.Path` and py._path.local.LocalPath
        objects.
    key : object, optional
        The group identifier in the store. Can be omitted if the HDF file
        contains a single pandas object.
    mode : {'r', 'r+', 'a'}, optional
        Mode to use when opening the file. Ignored if path_or_buf is a
        :class:`pandas.HDFStore`. Default is 'r'.
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
    .to_hdf : Write a HDF file from a DataFrame.
    """

    warnings.warn("Using CPU via Pandas to read HDF dataset, this may "
                  "be GPU accelerated in the future")
    pd_value = pd.read_hdf(
        path_or_buf,
        key=key,
        mode=mode,
        **kwargs
    )
    return cudf.from_pandas(pd_value)


def to_hdf(path_or_buf, key, value, mode=None, complevel=None, complib=None,
           append=None, **kwargs):
    """
    Write the contained data to an HDF5 file using HDFStore.

    Hierarchical Data Format (HDF) is self-describing, allowing an
    application to interpret the structure and contents of a file with
    no outside information. One HDF file can hold a mix of related objects
    which can be accessed as a group or as individual objects.

    In order to add another DataFrame or Series to an existing HDF file
    please use append mode and a different a key.

    For more information see the :ref:`user guide <io.hdf5>`.
    Parameters
    ----------
    path_or_buf : str or pandas.HDFStore
        File path or HDFStore object.
    key : str
        Identifier for the group in the store.
    mode : {'a', 'w', 'r+'}, default 'a'
        Mode to open file:
        - 'w': write, a new file is created (an existing file with
            the same name would be deleted).
        - 'a': append, an existing file is opened for reading and
            writing, and if the file does not exist it is created.
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
        of the object are indexed. See :ref:`io.hdf5-query-data-columns`.
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
    .read_hdf : Read from HDF file.
    """
    warnings.warn("Using CPU via Pandas to write HDF dataset, this may "
                  "be GPU accelerated in the future")
    print(value.index.name)
    pd_value = value.to_pandas()
    print(pd_value.index.name)
    pd.io.pytables.to_hdf(
        path_or_buf,
        key,
        pd_value,
        mode=mode,
        complevel=complevel,
        complib=complib,
        append=append,
        **kwargs
    )
