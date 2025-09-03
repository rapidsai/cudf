# Copyright (c) 2024-2025, NVIDIA CORPORATION.

from pylibcudf.libcudf cimport experimental as cpp_experimental


__all__ = ["disable", "disable_debugging", "enable", "enable_debugging"]

cpdef enable():
    """Turn on prefetching of managed memory."""
    cpp_experimental.enable()


cpdef disable():
    """Turn off prefetching of managed memory."""
    cpp_experimental.disable()


cpdef enable_debugging():
    """Enable prefetch debugging."""
    cpp_experimental.enable_debugging()


cpdef disable_debugging():
    """Disable prefetch debugging."""
    cpp_experimental.disable_debugging()
