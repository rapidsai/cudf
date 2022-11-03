# Copyright (c) 2019-2022, NVIDIA CORPORATION.

import warnings
from collections import abc
from io import BytesIO, StringIO

import numpy as np
import pandas as pd

import cudf
from cudf._lib import json as libjson
from cudf.api.types import is_list_like
from cudf.utils import ioutils
from cudf.utils.dtypes import _maybe_convert_to_default_type


@ioutils.doc_read_json()
def read_json(
    path_or_buf,
    engine="auto",
    orient=None,
    dtype=None,
    lines=False,
    compression="infer",
    byte_range=None,
    keep_quotes=False,
    storage_options=None,
    chunk_size=1024,
    *args,
    **kwargs,
):
    """{docstring}"""

    if dtype is not None and not isinstance(dtype, (abc.Mapping, bool)):
        raise TypeError(
            "'dtype' parameter only supports "
            "a dict of column names and types as key-value pairs, "
            f"or a bool, or None. Got {type(dtype)}"
        )

    if engine == "cudf" and not lines:
        raise ValueError(f"{engine} engine only supports JSON Lines format")
    if engine != "cudf_experimental" and keep_quotes:
        raise ValueError(
            "keep_quotes='True' is supported only with"
            " engine='cudf_experimental'"
        )
    if engine == "auto":
        engine = "cudf" if lines else "pandas"
    if (
        engine == "cudf"
        or engine == "cudf_experimental"
        or engine == "cudf_experimental_chunked"
    ):
        if dtype is None:
            dtype = True

        if kwargs:
            raise ValueError(
                "cudf engine doesn't support the "
                f"following keyword arguments: {list(kwargs.keys())}"
            )
        if args:
            raise ValueError(
                "cudf engine doesn't support the "
                f"following positional arguments: {list(args)}"
            )

        # Multiple sources are passed as a list. If a single source is passed,
        # wrap it in a list for unified processing downstream.
        if not is_list_like(path_or_buf):
            path_or_buf = [path_or_buf]

        filepaths_or_buffers = []
        for source in path_or_buf:
            if ioutils.is_directory(
                path_or_data=source, storage_options=storage_options
            ):
                fs = ioutils._ensure_filesystem(
                    passed_filesystem=None,
                    path=source,
                    storage_options=storage_options,
                )
                source = ioutils.stringify_pathlike(source)
                source = fs.sep.join([source, "*.json"])

            tmp_source, compression = ioutils.get_reader_filepath_or_buffer(
                path_or_data=source,
                compression=compression,
                iotypes=(BytesIO, StringIO),
                allow_raw_text_input=True,
                storage_options=storage_options,
            )
            if isinstance(tmp_source, list):
                filepaths_or_buffers.extend(tmp_source)
            else:
                filepaths_or_buffers.append(tmp_source)

        if engine == "cudf_experimental_chunked" and chunk_size != 0:
            df = chunked_read_json(
                filepaths_or_buffers,
                dtype,
                lines,
                compression,
                chunk_size,
                engine == "cudf_experimental_chunked",
                keep_quotes,
            )
        else:
            df = libjson.read_json(
                filepaths_or_buffers,
                dtype,
                lines,
                compression,
                byte_range,
                engine == "cudf_experimental",
                keep_quotes,
            )
    else:
        warnings.warn(
            "Using CPU via Pandas to read JSON dataset, this may "
            "be GPU accelerated in the future"
        )

        if not ioutils.ensure_single_filepath_or_buffer(
            path_or_data=path_or_buf,
            storage_options=storage_options,
        ):
            raise NotImplementedError(
                "`read_json` does not yet support reading "
                "multiple files via pandas"
            )

        path_or_buf, compression = ioutils.get_reader_filepath_or_buffer(
            path_or_data=path_or_buf,
            compression=compression,
            iotypes=(BytesIO, StringIO),
            allow_raw_text_input=True,
            storage_options=storage_options,
        )

        pd_value = pd.read_json(
            path_or_buf,
            lines=lines,
            dtype=dtype,
            compression=compression,
            storage_options=storage_options,
            orient=orient,
            *args,
            **kwargs,
        )
        df = cudf.from_pandas(pd_value)

    if dtype is None:
        dtype = True

    if dtype is True or isinstance(dtype, abc.Mapping):
        # There exists some dtypes in the result columns that is inferred.
        # Find them and map them to the default dtypes.
        dtype = {} if dtype is True else dtype
        unspecified_dtypes = {
            name: df._dtypes[name]
            for name in df._column_names
            if name not in dtype
        }
        default_dtypes = {}

        for name, dt in unspecified_dtypes.items():
            if dt == np.dtype("i1"):
                # csv reader reads all null column as int8.
                # The dtype should remain int8.
                default_dtypes[name] = dt
            else:
                default_dtypes[name] = _maybe_convert_to_default_type(dt)
        df = df.astype(default_dtypes)

    return df


def chunked_read_json(
    filepaths_or_buffers,
    dtype,
    lines,
    compression,
    chunk_size,
    is_experimental,
    keep_quotes,
):
    # find size of sources
    # compute num chunks
    # find first delim of each chunk.
    # compute record ranges
    # read record ranges
    # concat
    total_source_size = libjson.sources_size(
        filepaths_or_buffers, compression, [0, 0]
    )
    # print("total_source_size", total_source_size)
    num_chunks = (total_source_size + chunk_size - 1) // chunk_size
    delimiter_positions_in_chunks = [
        libjson.find_first_delimiter_in_chunk(
            filepaths_or_buffers,
            lines,
            compression,
            [chunk_index * chunk_size, chunk_size],
            is_experimental,
            "\n".encode()[0],
        )
        for chunk_index in range(num_chunks)
    ]
    delimiter_positions_in_chunks[0] = 0
    delimiter_positions_in_chunks = [
        pos + chunk_index * chunk_size
        for chunk_index, pos in zip(
            range(num_chunks), delimiter_positions_in_chunks
        )
        if pos is not None
    ]
    delimiter_positions_in_chunks.append(total_source_size)
    # print(delimiter_positions_in_chunks)
    record_ranges = list(
        zip(delimiter_positions_in_chunks, delimiter_positions_in_chunks[1:])
    )
    # print(record_ranges)

    # launch individual read_json chunked reads
    dfs = [
        libjson.read_json(
            filepaths_or_buffers,
            dtype,
            lines,
            compression,
            [chunk_start, chunk_end - chunk_start],
            is_experimental,
            keep_quotes,
        )
        for chunk_start, chunk_end in record_ranges
    ]

    # for i, df in enumerate(dfs):
    #     print("i=", i)
    #     print(df.columns, "\n", df.dtypes)
    #     print(df)
    # concat
    return cudf.concat(dfs, ignore_index=True)


@ioutils.doc_to_json()
def to_json(cudf_val, path_or_buf=None, *args, **kwargs):
    """{docstring}"""

    warnings.warn(
        "Using CPU via Pandas to write JSON dataset, this may "
        "be GPU accelerated in the future"
    )
    pd_value = cudf_val.to_pandas(nullable=True)
    return pd.io.json.to_json(path_or_buf, pd_value, *args, **kwargs)
