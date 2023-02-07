# Copyright (c) 2019-2023, NVIDIA CORPORATION.

from functools import partial

import dask

import cudf

from dask_cudf.backends import _default_backend


def read_json(url_path, engine="auto", **kwargs):
    """Read JSON data into a :class:`.DataFrame`.

    This function wraps :func:`dask.dataframe.read_json`, and passes
    ``engine=partial(cudf.read_json, engine="auto")`` by default.

    Parameters
    ----------
    url_path : str, list of str
        Location to read from. If a string, can include a glob character to
        find a set of file names.
        Supports protocol specifications such as ``"s3://"``.
    engine : str or Callable, default "auto"

        If str, this value will be used as the ``engine`` argument
        when :func:`cudf.read_json` is used to create each partition.
        If a :obj:`~typing.Callable`, this value will be used as the
        underlying function used to create each partition from JSON
        data. The default value is "auto", so that
        ``engine=partial(cudf.read_json, engine="auto")`` will be
        passed to :func:`dask.dataframe.read_json` by default.

    **kwargs :
        Key-word arguments to pass through to :func:`dask.dataframe.read_json`.

    Returns
    -------
    :class:`.DataFrame`

    Examples
    --------
    Load single file

    >>> from dask_cudf import read_json
    >>> read_json('myfile.json')  # doctest: +SKIP

    Load large line-delimited JSON files using partitions of approx
    256MB size

    >>> read_json('data/file*.csv', blocksize=2**28)  # doctest: +SKIP

    Load nested JSON data

    >>> read_json('myfile.json')  # doctest: +SKIP

    See Also
    --------
    dask.dataframe.read_json

    """

    # TODO: Add optimized code path to leverage the
    # `byte_range` argument in `cudf.read_json` for
    # local storage (see `dask_cudf.read_csv`)
    return _default_backend(
        dask.dataframe.read_json,
        url_path,
        engine=(
            partial(cudf.read_json, engine=engine)
            if isinstance(engine, str)
            else engine
        ),
        **kwargs,
    )
