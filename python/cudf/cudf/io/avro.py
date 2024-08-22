# Copyright (c) 2019-2024, NVIDIA CORPORATION.

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

    filepath_or_buffer = ioutils.get_reader_filepath_or_buffer(
        path_or_data=filepath_or_buffer,
        storage_options=storage_options,
    )
    filepath_or_buffer = ioutils._select_single_source(
        filepath_or_buffer, "read_avro"
    )

    return cudf.DataFrame._from_data(
        *libcudf.avro.read_avro(
            filepath_or_buffer, columns, skiprows, num_rows
        )
    )
