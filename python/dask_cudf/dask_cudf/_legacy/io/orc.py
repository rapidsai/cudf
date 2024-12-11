# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from io import BufferedWriter, IOBase

from fsspec.core import get_fs_token_paths
from fsspec.utils import stringify_path
from pyarrow import orc as orc

from dask import dataframe as dd
from dask.base import tokenize
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


def read_orc(path, columns=None, filters=None, storage_options=None, **kwargs):
    """Read ORC files into a :class:`.DataFrame`.

    Note that this function is mostly borrowed from upstream Dask.

    Parameters
    ----------
    path : str or list[str]
        Location of file(s), which can be a full URL with protocol specifier,
        and may include glob character if a single string.
    columns : None or list[str]
        Columns to load. If None, loads all.
    filters : None or list of tuple or list of lists of tuples
        If not None, specifies a filter predicate used to filter out
        row groups using statistics stored for each row group as
        Parquet metadata. Row groups that do not match the given
        filter predicate are not read. The predicate is expressed in
        `disjunctive normal form (DNF)
        <https://en.wikipedia.org/wiki/Disjunctive_normal_form>`__
        like ``[[('x', '=', 0), ...], ...]``. DNF allows arbitrary
        boolean logical combinations of single column predicates. The
        innermost tuples each describe a single column predicate. The
        list of inner predicates is interpreted as a conjunction
        (AND), forming a more selective and multiple column predicate.
        Finally, the outermost list combines these filters as a
        disjunction (OR). Predicates may also be passed as a list of
        tuples. This form is interpreted as a single conjunction. To
        express OR in predicates, one must use the (preferred)
        notation of list of lists of tuples.
    storage_options : None or dict
        Further parameters to pass to the bytes backend.

    See Also
    --------
    dask.dataframe.read_orc

    Returns
    -------
    dask_cudf.DataFrame

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
                f"Requested columns ({ex}) not in schema ({set(schema)})"
            )
    else:
        columns = list(schema)

    with fs.open(paths[0], "rb") as f:
        meta = cudf.read_orc(
            f,
            stripes=[0] if nstripes_per_file[0] else None,
            columns=columns,
            **kwargs,
        )

    name = "read-orc-" + tokenize(fs_token, path, columns, filters, **kwargs)
    dsk = {}
    N = 0
    for path, n in zip(paths, nstripes_per_file):
        for stripe in (
            range(n)
            if filters is None
            else cudf.io.orc._filter_stripes(filters, path)
        ):
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


def write_orc_partition(df, path, fs, filename, compression="snappy"):
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
    compression="snappy",
    compute=True,
    **kwargs,
):
    """
    Write a :class:`.DataFrame` to ORC file(s) (one file per partition).

    Parameters
    ----------
    df : DataFrame
    path : str or pathlib.Path
        Destination directory for data.  Prepend with protocol like ``s3://``
        or ``hdfs://`` for remote data.
    write_index : boolean, optional
        Whether or not to write the index. Defaults to True.
    storage_options : None or dict
        Further parameters to pass to the bytes backend.
    compression : string or dict, optional
    compute : bool, optional
        If True (default) then the result is computed immediately. If
        False then a :class:`~dask.delayed.Delayed` object is returned
        for future computation.

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
