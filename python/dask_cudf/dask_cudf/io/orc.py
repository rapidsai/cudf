import pyarrow.orc as orc

import dask.dataframe as dd
from dask.base import tokenize
from dask.bytes.core import get_fs_token_paths
from dask.dataframe.io.utils import _get_pyarrow_dtypes

import cudf


def _read_orc_stripe(fs, path, stripe, columns, kwargs={}):
    """Pull out specific columns from specific stripe"""
    with fs.open(path, "rb") as f:
        df_stripe = cudf.read_orc(f, stripe=stripe, columns=columns, **kwargs)
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
        meta = cudf.read_orc(f, stripe=0, columns=columns, **kwargs)

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
