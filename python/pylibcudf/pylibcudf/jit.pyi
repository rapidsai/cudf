# Copyright (c) 2025, NVIDIA CORPORATION.

from enum import IntEnum



class UDFSourceType(IntEnum):
    CUDA = ...
    PTX = ...


def is_runtime_jit_supported() -> bool: ...
