# Copyright (c) 2020, NVIDIA CORPORATION.

from cudf._libxx.cpp.io.types cimport source_info
from libcpp.string cimport string

import errno
from io import BytesIO, StringIO
import os


cdef source_info make_source_info(filepath_or_buffer) except*:
    cdef const unsigned char[::1] buf
    if isinstance(filepath_or_buffer, bytes):
        buf = filepath_or_buffer
    elif isinstance(filepath_or_buffer, BytesIO):
        buf = filepath_or_buffer.getbuffer()
    elif isinstance(filepath_or_buffer, StringIO):
        buf = filepath_or_buffer.read().encode()
    else:
        if os.path.isfile(filepath_or_buffer):
            return source_info(<string> str(filepath_or_buffer).encode())
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), filepath_or_buffer
        )
    return source_info(<char *>&buf[0], buf.shape[0])
