# Copyright (c) 2024-2025, NVIDIA CORPORATION.

from ..table cimport Table
from .types cimport Stream


cpdef Table make_timezone_transition_table(
    str tzif_dir, str timezone_name, Stream stream=*
)
