# Copyright (c) 2022-2024, NVIDIA CORPORATION.
from libcpp cimport bool
from libcpp.string cimport string
from pylibcudf.exception_handler cimport libcudf_exception_handler


cdef extern from "cudf/utilities/prefetch.hpp" \
        namespace "cudf::experimental::prefetch" nogil:
    # Not technically the right signature, but it's good enough to let Cython
    # generate valid C++ code. It just means we'll be copying a host string
    # extra, but that's OK. If we care we could generate string_view bindings,
    # but there's no real rush so if we go that route we might as well
    # contribute them upstream to Cython itself.
    void enable_prefetching(string key)
    void disable_prefetching(string key)
    void prefetch_debugging(bool enable)
