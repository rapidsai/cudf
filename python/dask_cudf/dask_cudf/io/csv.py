# Copyright (c) 2020, NVIDIA CORPORATION.

import os
from glob import glob
from warnings import warn

from dask import dataframe as dd
from dask.base import tokenize
from dask.compatibility import apply
from dask.dataframe.io.csv import make_reader
from dask.utils import parse_bytes

import cudf


def read_csv(path, chunksize="256 MiB", **kwargs):
    if "://" in str(path):
        func = make_reader(cudf.read_csv, "read_csv", "CSV")
        return func(path, blocksize=chunksize, **kwargs)
    else:
        return _internal_read_csv(path=path, chunksize=chunksize, **kwargs)


def _internal_read_csv(path, chunksize="256 MiB", **kwargs):
    if isinstance(chunksize, str):
        chunksize = parse_bytes(chunksize)

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

    compression = kwargs.get("compression", False)
    if compression and chunksize:
        # compressed CSVs reading must read the entire file
        kwargs.pop("byte_range", None)
        warn(
            "Warning %s compression does not support breaking apart files\n"
            "Please ensure that each individual file can fit in memory and\n"
            "use the keyword ``chunksize=None to remove this message``\n"
            "Setting ``chunksize=(size of file)``" % compression
        )
        chunksize = None

    if chunksize is None:
        return read_csv_without_chunksize(path, **kwargs)

    dask_reader = make_reader(cudf.read_csv, "read_csv", "CSV")
    meta = dask_reader(filenames[0], **kwargs)._meta

    dsk = {}
    i = 0
    dtypes = meta.dtypes.values

    for fn in filenames:
        size = os.path.getsize(fn)
        for start in range(0, size, chunksize):
            kwargs2 = kwargs.copy()
            kwargs2["byte_range"] = (
                start,
                chunksize,
            )  # specify which chunk of the file we care about
            if start != 0:
                kwargs2[
                    "names"
                ] = meta.columns  # no header in the middle of the file
                kwargs2["header"] = None
            dsk[(name, i)] = (apply, _read_csv, [fn, dtypes], kwargs2)

            i += 1

    divisions = [None] * (len(dsk) + 1)
    return dd.core.new_dd_object(dsk, name, meta, divisions)


def _read_csv(fn, dtypes=None, **kwargs):
    return cudf.read_csv(fn, **kwargs)


def read_csv_without_chunksize(path, **kwargs):
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

    # Read "head" of first file (first 5 rows).
    # Convert to empty df for metadata.
    meta = cudf.read_csv(filenames[0], nrows=5, **kwargs).iloc[:0]

    graph = {
        (name, i): (apply, cudf.read_csv, [fn], kwargs)
        for i, fn in enumerate(filenames)
    }

    divisions = [None] * (len(filenames) + 1)

    return dd.core.new_dd_object(graph, name, meta, divisions)
