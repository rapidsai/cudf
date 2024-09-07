# Copyright (c) 2024, NVIDIA CORPORATION.

from ..table cimport Table


cpdef Table make_timezone_transition_table(tzif_dir, str timezone_name)
