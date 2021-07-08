# Copyright (c) 2019, NVIDIA CORPORATION.
import cudf
from cudf import _lib as libcudf
from cudf.utils import ioutils


@ioutils.doc_read_avro()
def read_avro(
    filepath_or_buffer,
    engine="cudf",
    columns=None,
    skiprows=None,
    num_rows=None,
    **kwargs,
):
    """{docstring}"""

    is_single_filepath_or_buffer = ioutils.ensure_single_filepath_or_buffer(
        path_or_data=filepath_or_buffer, **kwargs,
    )
    if not is_single_filepath_or_buffer:
        raise NotImplementedError(
            "`read_avro` does not yet support reading multiple files"
        )

    filepath_or_buffer, compression = ioutils.get_filepath_or_buffer(
        path_or_data=filepath_or_buffer, compression=None, **kwargs
    )
    if compression is not None:
        ValueError("URL content-encoding decompression is not supported")

    if engine == "cudf":
        return cudf.DataFrame._from_data(
            *libcudf.avro.read_avro(
                filepath_or_buffer, columns, skiprows, num_rows
            )
        )
    else:
        raise NotImplementedError("read_avro currently only supports cudf")
