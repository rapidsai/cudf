# Copyright (c) 2024-2025, NVIDIA CORPORATION.
from pylibcudf.libcudf.strings.side_type import \
    side_type as SideType  # no-cython-lint

__all__ = ["SideType"]

SideType.__str__ = SideType.__repr__
