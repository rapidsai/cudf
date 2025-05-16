# Copyright (c) 2024, NVIDIA CORPORATION.

from libcpp cimport bool
from libcpp.string cimport string
from pylibcudf.libcudf cimport experimental as cpp_experimental


__all__ = ["disable_prefetching", "enable_prefetching", "prefetch_debugging"]

cpdef enable_prefetching(str key):
    """Turn on prefetch instructions for the given key.

    Parameters
    ----------
    key : str
        The key to enable prefetching for.
    """
    cdef string c_key = key.encode("utf-8")
    cpp_experimental.enable_prefetching(c_key)


cpdef disable_prefetching(str key):
    """Turn off prefetch instructions for the given key.

    Parameters
    ----------
    key : str
        The key to disable prefetching for.
    """
    cdef string c_key = key.encode("utf-8")
    cpp_experimental.disable_prefetching(c_key)


cpdef prefetch_debugging(bool enable):
    """Enable or disable prefetch debugging.

    When enabled, any prefetch instructions will be logged to the console.

    Parameters
    ----------
    enable : bool
        Whether to enable or disable prefetch debugging.
    """
    cpp_experimental.prefetch_debugging(enable)
