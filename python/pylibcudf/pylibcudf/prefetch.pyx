# Copyright (c) 2024-2025, NVIDIA CORPORATION.

from pylibcudf.libcudf cimport prefetch as cpp_prefetch


__all__ = ["disable", "disable_debugging", "enable", "enable_debugging"]

cpdef enable():
    """Turn on prefetching of managed memory."""
    cpp_prefetch.enable()


cpdef disable():
    """Turn off prefetching of managed memory."""
    cpp_prefetch.disable()


cpdef enable_debugging():
    """Enable prefetch debugging."""
    cpp_prefetch.enable_debugging()


cpdef disable_debugging():
    """Disable prefetch debugging."""
    cpp_prefetch.disable_debugging()
