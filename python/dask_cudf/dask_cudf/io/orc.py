# Copyright (c) 2020, NVIDIA CORPORATION.

from io import BufferedWriter, IOBase

from pyarrow import orc as orc

from dask import dataframe as dd
from dask.base import tokenize
from dask.bytes.core import get_fs_token_paths, stringify_path
from dask.dataframe.io.utils import _get_pyarrow_dtypes

import cudf


def _read_orc_stripe(fs, path, stripe, columns, kwargs=None):
    """Pull out specific columns from specific stripe"""
    if kwargs is None:
        kwargs = {}
    with fs.open(path, "rb") as f:
        df_stripe = cudf.read_orc(
            f, stripes=[stripe], columns=columns, **kwargs
        )
    return df_stripe


def read_orc(path, columns=None, storage_options=None, **kwargs):
    """Read cudf dataframe from ORC file(s).

    Note that this function is mostly borrowed from upstream Dask.

    Parameters
    ----------
    path: str or list(str)
        Location of file(s), which can be a full URL with protocol specifier,
        and may include glob character if a single string.
    columns: None or list(str)
        Columns to load. If None, loads all.
    storage_options: None or dict
        Further parameters to pass to the bytes backend.

    Returns
    -------
    cudf.DataFrame
    """

    storage_options = storage_options or {}
    fs, fs_token, paths = get_fs_token_paths(
        path, mode="rb", storage_options=storage_options
    )
    schema = None
    nstripes_per_file = []
    for path in paths:
        with fs.open(path, "rb") as f:
            o = orc.ORCFile(f)
            if schema is None:
                schema = o.schema
            elif schema != o.schema:
                raise ValueError(
                    "Incompatible schemas while parsing ORC files"
                )
            nstripes_per_file.append(o.nstripes)
    schema = _get_pyarrow_dtypes(schema, categories=None)
    if columns is not None:
        ex = set(columns) - set(schema)
        if ex:
            raise ValueError(
                "Requested columns (%s) not in schema (%s)" % (ex, set(schema))
            )
    else:
        columns = list(schema)

    with fs.open(paths[0], "rb") as f:
        meta = cudf.read_orc(f, stripes=[0], columns=columns, **kwargs)

    name = "read-orc-" + tokenize(fs_token, path, columns, **kwargs)
    dsk = {}
    N = 0
    for path, n in zip(paths, nstripes_per_file):
        for stripe in range(n):
            dsk[(name, N)] = (
                _read_orc_stripe,
                fs,
                path,
                stripe,
                columns,
                kwargs,
            )
            N += 1

    divisions = [None] * (len(dsk) + 1)
    return dd.core.new_dd_object(dsk, name, meta, divisions)


def write_orc_partition(df, path, fs, filename, compression=None):
    full_path = fs.sep.join([path, filename])
    with fs.open(full_path, mode="wb") as out_file:
        if not isinstance(out_file, IOBase):
            out_file = BufferedWriter(out_file)
        cudf.io.to_orc(df, out_file, compression=compression)
    return full_path


def to_orc(
    df,
    path,
    write_index=True,
    storage_options=None,
    compression=None,
    compute=True,
    **kwargs,
):
    """Write a dask_cudf dataframe to ORC file(s) (one file per partition).

    Parameters
    ----------
    df : dask_cudf.DataFrame
    path: string or pathlib.Path
        Destination directory for data.  Prepend with protocol like ``s3://``
        or ``hdfs://`` for remote data.
    write_index : boolean, optional
        Whether or not to write the index. Defaults to True.
    storage_options: None or dict
        Further parameters to pass to the bytes backend.
    compression : string or dict, optional
    compute : bool, optional
        If True (default) then the result is computed immediately. If False
        then a ``dask.delayed`` object is returned for future computation.
    """

    from dask import compute as dask_compute, delayed

    # TODO: Use upstream dask implementation once available
    #       (see: Dask Issue#5596)

    if hasattr(path, "name"):
        path = stringify_path(path)
    fs, _, _ = get_fs_token_paths(
        path, mode="wb", storage_options=storage_options
    )
    # Trim any protocol information from the path before forwarding
    path = fs._strip_protocol(path)

    if write_index:
        df = df.reset_index()
    else:
        # Not writing index - might as well drop it
        df = df.reset_index(drop=True)

    fs.mkdirs(path, exist_ok=True)

    # Use i_offset and df.npartitions to define file-name list
    filenames = ["part.%i.orc" % i for i in range(df.npartitions)]

    # write parts
    dwrite = delayed(write_orc_partition)
    parts = [
        dwrite(d, path, fs, filename, compression=compression)
        for d, filename in zip(df.to_delayed(), filenames)
    ]

    if compute:
        return dask_compute(*parts)

    return delayed(list)(parts)
