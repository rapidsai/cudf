# Copyright (c) 2022-2024, NVIDIA CORPORATION.

import os
from glob import glob

import dask.dataframe as dd
from dask.base import tokenize
from dask.utils import apply, parse_bytes

import cudf


def read_text(path, chunksize="256 MiB", **kwargs):
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

    name = "read-text-" + tokenize(path, tokenize, **kwargs)

    if chunksize:
        dsk = {}
        i = 0
        for fn in filenames:
            size = os.path.getsize(fn)
            for start in range(0, size, chunksize):
                kwargs1 = kwargs.copy()
                kwargs1["byte_range"] = (
                    start,
                    chunksize,
                )  # specify which chunk of the file we care about

                dsk[(name, i)] = (apply, cudf.read_text, [fn], kwargs1)
                i += 1
    else:
        dsk = {
            (name, i): (apply, cudf.read_text, [fn], kwargs)
            for i, fn in enumerate(filenames)
        }

    meta = cudf.Series([], dtype="O")
    divisions = [None] * (len(dsk) + 1)
    return dd.core.new_dd_object(dsk, name, meta, divisions)
