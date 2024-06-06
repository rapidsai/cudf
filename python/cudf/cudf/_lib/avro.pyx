# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from cudf._lib.utils cimport data_from_pylibcudf_io

import cudf._lib.pylibcudf as plc
from cudf._lib.pylibcudf.io.types import SourceInfo


cpdef read_avro(datasource, columns=None, skip_rows=0, num_rows=-1):
    """
    Cython function to call libcudf read_avro, see `read_avro`.

    See Also
    --------
    cudf.io.avro.read_avro
    """

    num_rows = -1 if num_rows is None else num_rows
    skip_rows = 0 if skip_rows is None else skip_rows

    if not isinstance(num_rows, int) or num_rows < -1:
        raise TypeError("num_rows must be an int >= -1")
    if not isinstance(skip_rows, int) or skip_rows < 0:
        raise TypeError("skip_rows must be an int >= 0")

    return data_from_pylibcudf_io(
        plc.io.avro.read_avro(
            SourceInfo([datasource]),
            columns,
            skip_rows,
            num_rows
        )
    )
