# Copyright (c) 2024-2025, NVIDIA CORPORATION.
from libc.stdint cimport int32_t
from pylibcudf.libcudf.round cimport rounding_method

from .column cimport Column

from rmm.pylibrmm.stream cimport Stream


cpdef Column round(
    Column source,
    int32_t decimal_places = *,
    rounding_method round_method = *,
    Stream stream = *
)
