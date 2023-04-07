# Copyright (c) 2019-2022, NVIDIA CORPORATION.

import cudf
from cudf import _lib as libcudf
from cudf.utils import ioutils


@ioutils.doc_read_avro()
def read_avro(
    filepath_or_buffer,
    columns=None,
    skiprows=None,
    num_rows=None,
    storage_options=None,
):
    """{docstring}"""

    is_single_filepath_or_buffer = ioutils.ensure_single_filepath_or_buffer(
        path_or_data=filepath_or_buffer,
        storage_options=storage_options,
    )
    if not is_single_filepath_or_buffer:
        raise NotImplementedError(
            "`read_avro` does not yet support reading multiple files"
        )

    filepath_or_buffer, compression = ioutils.get_reader_filepath_or_buffer(
        path_or_data=filepath_or_buffer,
        compression=None,
        storage_options=storage_options,
    )
    if compression is not None:
        ValueError("URL content-encoding decompression is not supported")

    return cudf.DataFrame._from_data(
        *libcudf.avro.read_avro(
            filepath_or_buffer, columns, skiprows, num_rows
        )
    )
