# Copyright (c) 2019-2024, NVIDIA CORPORATION.

from functools import partial

import numpy as np
from fsspec.core import get_compression, get_fs_token_paths

import dask
from dask.utils import parse_bytes

import cudf
from cudf.core.column import as_column
from cudf.utils.ioutils import _is_local_filesystem

from dask_cudf.backends import _default_backend


def _read_json_partition(
    paths,
    fs=None,
    include_path_column=False,
    path_converter=None,
    **kwargs,
):
    # Transfer all data up front for remote storage
    sources = (
        paths
        if fs is None
        else fs.cat_ranges(
            paths,
            [0] * len(paths),
            fs.sizes(paths),
        )
    )

    if include_path_column:
        # Add "path" column.
        # Must iterate over sources sequentially
        if not isinstance(include_path_column, str):
            include_path_column = "path"
        converted_paths = (
            paths
            if path_converter is None
            else [path_converter(path) for path in paths]
        )
        dfs = []
        for i, source in enumerate(sources):
            df = cudf.read_json(source, **kwargs)
            df[include_path_column] = as_column(
                converted_paths[i], length=len(df)
            )
            dfs.append(df)
        return cudf.concat(dfs)
    else:
        # Pass sources directly to cudf
        return cudf.read_json(sources, **kwargs)


def read_json(
    url_path,
    engine="auto",
    blocksize=None,
    orient="records",
    lines=None,
    compression="infer",
    aggregate_files=True,
    **kwargs,
):
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
        If a :obj:`~collections.abc.Callable`, this value will be used as the
        underlying function used to create each partition from JSON
        data. The default value is "auto", so that
        ``engine=partial(cudf.read_json, engine="auto")`` will be
        passed to :func:`dask.dataframe.read_json` by default.
    aggregate_files : bool or int
        Whether to map multiple files to each output partition. If True,
        the `blocksize` argument will be used to determine the number of
        files in each partition. If any one file is larger than `blocksize`,
        the `aggregate_files` argument will be ignored. If an integer value
        is specified, the `blocksize` argument will be ignored, and that
        number of files will be mapped to each partition. Default is True.
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

    if lines is None:
        lines = orient == "records"
    if orient != "records" and lines:
        raise ValueError(
            'Line-delimited JSON is only available with orient="records".'
        )
    if blocksize and (orient != "records" or not lines):
        raise ValueError(
            "JSON file chunking only allowed for JSON-lines"
            "input (orient='records', lines=True)."
        )

    inputs = []
    if aggregate_files and blocksize or int(aggregate_files) > 1:
        # Attempt custom read if we are mapping multiple files
        # to each output partition. Otherwise, upstream logic
        # is sufficient.

        storage_options = kwargs.get("storage_options", {})
        fs, _, paths = get_fs_token_paths(
            url_path, mode="rb", storage_options=storage_options
        )
        if isinstance(aggregate_files, int) and aggregate_files > 1:
            # Map a static file count to each partition
            inputs = [
                paths[offset : offset + aggregate_files]
                for offset in range(0, len(paths), aggregate_files)
            ]
        elif aggregate_files is True and blocksize:
            # Map files dynamically (using blocksize)
            file_sizes = fs.sizes(paths)  # NOTE: This can be slow
            blocksize = parse_bytes(blocksize)
            if all([file_size <= blocksize for file_size in file_sizes]):
                counts = np.unique(
                    np.floor(np.cumsum(file_sizes) / blocksize),
                    return_counts=True,
                )[1]
                offsets = np.concatenate([[0], counts.cumsum()])
                inputs = [
                    paths[offsets[i] : offsets[i + 1]]
                    for i in range(len(offsets) - 1)
                ]

    if inputs:
        # Inputs were successfully populated.
        # Use custom _read_json_partition function
        # to generate each partition.

        compression = get_compression(
            url_path[0] if isinstance(url_path, list) else url_path,
            compression,
        )
        _kwargs = dict(
            orient=orient,
            lines=lines,
            compression=compression,
            include_path_column=kwargs.get("include_path_column", False),
            path_converter=kwargs.get("path_converter"),
        )
        if not _is_local_filesystem(fs):
            _kwargs["fs"] = fs
        # TODO: Generate meta more efficiently
        meta = _read_json_partition(inputs[0][:1], **_kwargs)
        return dask.dataframe.from_map(
            _read_json_partition,
            inputs,
            meta=meta,
            **_kwargs,
        )

    # Fall back to dask.dataframe.read_json
    return _default_backend(
        dask.dataframe.read_json,
        url_path,
        engine=(
            partial(cudf.read_json, engine=engine)
            if isinstance(engine, str)
            else engine
        ),
        blocksize=blocksize,
        orient=orient,
        lines=lines,
        compression=compression,
        **kwargs,
    )
