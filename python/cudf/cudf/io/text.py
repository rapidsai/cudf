# Copyright (c) 2018-2021, NVIDIA CORPORATION.

from io import BytesIO, StringIO

from nvtx import annotate

import cudf
from cudf._lib import text as libtext
from cudf.utils import ioutils
from cudf.utils.dtypes import is_list_like


@annotate("READ_TEXT", color="purple", domain="cudf_python")
@ioutils.doc_read_text()
def read_text(
    filepath_or_buffer, delimiter=None, compression="infer", **kwargs,
):
    """{docstring}"""

    # Multiple sources are passed as a list. If a single source is passed,
    # wrap it in a list for unified processing downstream.
    if not is_list_like(filepath_or_buffer):
        filepath_or_buffer = [filepath_or_buffer]

    filepath_or_buffer, compression = ioutils.get_filepath_or_buffer(
        path_or_data=filepath_or_buffer,
        compression=compression,
        iotypes=(BytesIO, StringIO),
        **kwargs,
    )

    df = cudf.DataFrame._from_table(
        libtext.read_text(
            filepath_or_buffer, delimiter=delimiter, compression=compression,
        )
    )

    print(df.head())

    return df
