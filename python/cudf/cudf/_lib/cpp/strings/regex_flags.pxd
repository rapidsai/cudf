# Copyright (c) 2022, NVIDIA CORPORATION.

cdef extern from "cudf/strings/regex/flags.hpp" \
        namespace "cudf::strings" nogil:

    ctypedef enum regex_flags:
        DEFAULT 'cudf::strings::regex_flags::DEFAULT'
        MULTILINE  'cudf::strings::regex_flags::MULTILINE'
        DOTALL 'cudf::strings::regex_flags::DOTALL'

    ctypedef enum capture_groups:
        EXTRACT 'cudf::strings::capture_groups::EXTRACT'
        NON_CAPTURE 'cudf::strings::capture_groups::NON_CAPTURE'
