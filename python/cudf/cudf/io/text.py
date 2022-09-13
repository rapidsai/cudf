# Copyright (c) 2018-2022, NVIDIA CORPORATION.

from io import BytesIO, StringIO

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
    **kwargs,
):
    """{docstring}"""

    filepath_or_buffer, compression = ioutils.get_reader_filepath_or_buffer(
        path_or_data=filepath_or_buffer,
        compression=None,
        iotypes=(BytesIO, StringIO),
        **kwargs,
    )

    return cudf.Series._from_data(
        libtext.read_text(
            filepath_or_buffer, delimiter=delimiter, byte_range=byte_range
        )
    )
