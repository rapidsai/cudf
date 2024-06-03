# Copyright (c) 2024, NVIDIA CORPORATION.
from libc.stdint cimport int32_t

from cudf._lib.pylibcudf.libcudf.round cimport rounding_method

from .column cimport Column


cpdef Column round(
    Column source,
    int32_t decimal_places = *,
    rounding_method round_method = *
)
