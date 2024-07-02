# Copyright (c) 2018-2024, NVIDIA CORPORATION.

from io import BytesIO, StringIO

import cudf
from cudf._lib import text as libtext
from cudf.utils import ioutils
from cudf.utils.performance_tracking import _performance_tracking


@_performance_tracking
@ioutils.doc_read_text()
def read_text(
    filepath_or_buffer,
    delimiter=None,
    byte_range=None,
    strip_delimiters=False,
    compression=None,
    compression_offsets=None,
    storage_options=None,
    prefetch_read_ahead=None,
):
    """{docstring}"""

    if delimiter is None:
        raise ValueError("delimiter needs to be provided")

    # Extract filesystem up front
    fs, paths = ioutils._get_filesystem_and_paths(
        path_or_data=filepath_or_buffer, storage_options=storage_options
    )

    # Prefetch remote data if possible
    if fs and paths:
        filepath_or_buffer, info = ioutils.prefetch_remote_buffers(
            paths,
            fs,
            prefetcher="contiguous",
            prefetcher_options={
                "byte_range": byte_range,
                "read_ahead": prefetch_read_ahead,
            },
        )
        assert len(filepath_or_buffer) == 1
        filepath_or_buffer = filepath_or_buffer[0]
        byte_range = info.get("byte_range", byte_range)

    filepath_or_buffer, _ = ioutils.get_reader_filepath_or_buffer(
        path_or_data=filepath_or_buffer,
        compression=None,
        fs=fs,
        iotypes=(BytesIO, StringIO),
        storage_options=storage_options,
    )

    return cudf.Series._from_data(
        libtext.read_text(
            filepath_or_buffer,
            delimiter=delimiter,
            byte_range=byte_range,
            strip_delimiters=strip_delimiters,
            compression=compression,
            compression_offsets=compression_offsets,
        )
    )
