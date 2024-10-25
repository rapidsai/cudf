# Copyright (c) 2020-2023, NVIDIA CORPORATION.

import os
from glob import glob
from warnings import warn

from fsspec.utils import infer_compression

from dask import dataframe as dd
from dask.base import tokenize
from dask.dataframe.io.csv import make_reader
from dask.utils import apply, parse_bytes

import cudf


def read_csv(path, blocksize="default", **kwargs):
    """
    Read CSV files into a :class:`.DataFrame`.

    This API parallelizes the :func:`cudf:cudf.read_csv` function in
    the following ways:

    It supports loading many files at once using globstrings:

    >>> import dask_cudf
    >>> df = dask_cudf.read_csv("myfiles.*.csv")

    In some cases it can break up large files:

    >>> df = dask_cudf.read_csv("largefile.csv", blocksize="256 MiB")

    It can read CSV files from external resources (e.g. S3, HTTP, FTP)

    >>> df = dask_cudf.read_csv("s3://bucket/myfiles.*.csv")
    >>> df = dask_cudf.read_csv("https://www.mycloud.com/sample.csv")

    Internally ``read_csv`` uses :func:`cudf:cudf.read_csv` and
    supports many of the same keyword arguments with the same
    performance guarantees. See the docstring for
    :func:`cudf:cudf.read_csv` for more information on available
    keyword arguments.

    Parameters
    ----------
    path : str, path object, or file-like object
        Either a path to a file (a str, :py:class:`pathlib.Path`, or
        py._path.local.LocalPath), URL (including http, ftp, and S3
        locations), or any object with a read() method (such as
        builtin :py:func:`open` file handler function or
        :py:class:`~io.StringIO`).
    blocksize : int or str, default "256 MiB"
        The target task partition size. If ``None``, a single block
        is used for each file.
    **kwargs : dict
        Passthrough key-word arguments that are sent to
        :func:`cudf:cudf.read_csv`.

    Notes
    -----
    If any of `skipfooter`/`skiprows`/`nrows` are passed,
    `blocksize` will default to None.

    Examples
    --------
    >>> import dask_cudf
    >>> ddf = dask_cudf.read_csv("sample.csv", usecols=["a", "b"])
    >>> ddf.compute()
       a      b
    0  1     hi
    1  2  hello
    2  3     ai

    """

    # Handle `chunksize` deprecation
    if "chunksize" in kwargs:
        chunksize = kwargs.pop("chunksize", "default")
        warn(
            "`chunksize` is deprecated and will be removed in the future. "
            "Please use `blocksize` instead.",
            FutureWarning,
        )
        if blocksize == "default":
            blocksize = chunksize

    # Set default `blocksize`
    if blocksize == "default":
        if (
            kwargs.get("skipfooter", 0) != 0
            or kwargs.get("skiprows", 0) != 0
            or kwargs.get("nrows", None) is not None
        ):
            # Cannot read in blocks if skipfooter,
            # skiprows or nrows is passed.
            blocksize = None
        else:
            blocksize = "256 MiB"

    if "://" in str(path):
        func = make_reader(cudf.read_csv, "read_csv", "CSV")
        return func(path, blocksize=blocksize, **kwargs)
    else:
        return _internal_read_csv(path=path, blocksize=blocksize, **kwargs)


def _internal_read_csv(path, blocksize="256 MiB", **kwargs):
    if isinstance(blocksize, str):
        blocksize = parse_bytes(blocksize)

    if isinstance(path, list):
        filenames = path
    elif isinstance(path, str):
        filenames = sorted(glob(path))
    elif hasattr(path, "__fspath__"):
        filenames = sorted(glob(path.__fspath__()))
    else:
        raise TypeError(f"Path type not understood:{type(path)}")

    if not filenames:
        msg = f"A file in: {filenames} does not exist."
        raise FileNotFoundError(msg)

    name = "read-csv-" + tokenize(
        path, tokenize, **kwargs
    )  # TODO: get last modified time

    compression = kwargs.get("compression", "infer")

    if compression == "infer":
        # Infer compression from first path by default
        compression = infer_compression(filenames[0])

    if compression and blocksize:
        # compressed CSVs reading must read the entire file
        kwargs.pop("byte_range", None)
        warn(
            "Warning %s compression does not support breaking apart files\n"
            "Please ensure that each individual file can fit in memory and\n"
            "use the keyword ``blocksize=None to remove this message``\n"
            "Setting ``blocksize=(size of file)``" % compression
        )
        blocksize = None

    if blocksize is None:
        return read_csv_without_blocksize(path, **kwargs)

    # Let dask.dataframe generate meta
    dask_reader = make_reader(cudf.read_csv, "read_csv", "CSV")
    kwargs1 = kwargs.copy()
    usecols = kwargs1.pop("usecols", None)
    dtype = kwargs1.pop("dtype", None)
    meta = dask_reader(filenames[0], **kwargs1)._meta
    names = meta.columns
    if usecols or dtype:
        # Regenerate meta with original kwargs if
        # `usecols` or `dtype` was specified
        meta = dask_reader(filenames[0], **kwargs)._meta

    dsk = {}
    i = 0
    dtypes = meta.dtypes.values

    for fn in filenames:
        size = os.path.getsize(fn)
        for start in range(0, size, blocksize):
            kwargs2 = kwargs.copy()
            kwargs2["byte_range"] = (
                start,
                blocksize,
            )  # specify which chunk of the file we care about
            if start != 0:
                kwargs2["names"] = names  # no header in the middle of the file
                kwargs2["header"] = None
            dsk[(name, i)] = (apply, _read_csv, [fn, dtypes], kwargs2)

            i += 1

    divisions = [None] * (len(dsk) + 1)
    return dd.core.new_dd_object(dsk, name, meta, divisions)


def _read_csv(fn, dtypes=None, **kwargs):
    return cudf.read_csv(fn, **kwargs)


def read_csv_without_blocksize(path, **kwargs):
    """Read entire CSV with optional compression (gzip/zip)

    Parameters
    ----------
    path : str
        path to files (support for glob)
    """
    if isinstance(path, list):
        filenames = path
    elif isinstance(path, str):
        filenames = sorted(glob(path))
    elif hasattr(path, "__fspath__"):
        filenames = sorted(glob(path.__fspath__()))
    else:
        raise TypeError(f"Path type not understood:{type(path)}")

    name = "read-csv-" + tokenize(path, **kwargs)

    meta_kwargs = kwargs.copy()
    if "skipfooter" in meta_kwargs:
        meta_kwargs.pop("skipfooter")
    if "nrows" in meta_kwargs:
        meta_kwargs.pop("nrows")
    # Read "head" of first file (first 5 rows).
    # Convert to empty df for metadata.
    meta = cudf.read_csv(filenames[0], nrows=5, **meta_kwargs).iloc[:0]

    graph = {
        (name, i): (apply, cudf.read_csv, [fn], kwargs)
        for i, fn in enumerate(filenames)
    }

    divisions = [None] * (len(filenames) + 1)

    return dd.core.new_dd_object(graph, name, meta, divisions)
