# Copyright (c) 2022-2024, NVIDIA CORPORATION.

import os
from glob import glob

import dask.dataframe as dd
from dask.utils import parse_bytes

import cudf


def _read_text(source, **kwargs):
    # Wrapper for cudf.read_text operation
    fn, byte_range = source
    return cudf.read_text(fn, byte_range=byte_range, **kwargs)


def read_text(path, chunksize="256 MiB", byte_range=None, **kwargs):
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

    if chunksize and byte_range:
        raise ValueError("Cannot specify both chunksize and byte_range.")

    if chunksize:
        sources = []
        for fn in filenames:
            size = os.path.getsize(fn)
            for start in range(0, size, chunksize):
                byte_range = (
                    start,
                    chunksize,
                )  # specify which chunk of the file we care about
                sources.append((fn, byte_range))
    else:
        sources = [(fn, byte_range) for fn in filenames]

    return dd.from_map(
        _read_text,
        sources,
        meta=cudf.Series([], dtype="O"),
        **kwargs,
    )
