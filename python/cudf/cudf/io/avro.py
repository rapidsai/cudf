# Copyright (c) 2019, NVIDIA CORPORATION.
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

    from cudf import DataFrame

    filepath_or_buffer, compression = ioutils.get_filepath_or_buffer(
        path_or_data=filepath_or_buffer, compression=None, **kwargs
    )
    if compression is not None:
        ValueError("URL content-encoding decompression is not supported")

    if engine == "cudf":
        return DataFrame._from_table(
            libcudf.avro.read_avro(
                filepath_or_buffer, columns, skiprows, num_rows
            )
        )
    else:
        raise NotImplementedError("read_avro currently only supports cudf")
