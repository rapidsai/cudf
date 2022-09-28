# Copyright (c) 2018-2022, NVIDIA CORPORATION.

from io import BytesIO, StringIO, TextIOBase

import cudf
from cudf._lib import text as libtext
from cudf.utils import ioutils
from cudf.utils.utils import _cudf_nvtx_annotate


@_cudf_nvtx_annotate
@ioutils.doc_read_text()
def read_text(
    filepath_or_buffer,
    delimiter=None,
    byte_range=None,
    compression=None,
    compression_offsets=None,
    **kwargs,
):
    """{docstring}"""

    filepath_or_buffer, _ = ioutils.get_reader_filepath_or_buffer(
        path_or_data=filepath_or_buffer,
        compression=None,
        iotypes=(BytesIO, StringIO),
        **kwargs,
    )

    if compression == "bgzip":
        if isinstance(filepath_or_buffer, TextIOBase):
            raise ValueError("bgzip compression requires a file path")
        filepath_or_buffer = libtext.BGZIPFile(
            filepath_or_buffer, compression_offsets
        )
    elif compression is not None:
        raise ValueError("Only bgzip compression is supported at the moment")
    elif compression_offsets is not None:
        raise ValueError("compression_offsets requires compression to be set")

    return cudf.Series._from_data(
        libtext.read_text(
            filepath_or_buffer, delimiter=delimiter, byte_range=byte_range
        )
    )
