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
):
    """{docstring}"""

    if delimiter is None:
        raise ValueError("delimiter needs to be provided")

    filepath_or_buffer = ioutils.get_reader_filepath_or_buffer(
        path_or_data=filepath_or_buffer,
        iotypes=(BytesIO, StringIO),
        storage_options=storage_options,
    )
    filepath_or_buffer = ioutils._select_single_source(
        filepath_or_buffer, "read_text"
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
