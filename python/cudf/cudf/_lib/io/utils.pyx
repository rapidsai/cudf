# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp.string cimport string
from cudf._lib.cpp.io.types cimport source_info

import errno
import io
import os

# Converts the Python source input to libcudf++ IO source_info
# with the appropriate type and source values
cdef source_info make_source_info(src) except*:
    cdef const unsigned char[::1] buf
    if isinstance(src, bytes):
        buf = src
    elif isinstance(src, io.BytesIO):
        buf = src.getbuffer()
    # Otherwise src is expected to be a numeric fd, string path, or PathLike.
    # TODO (ptaylor): Might need to update this check if accepted input types
    #                 change when UCX and/or cuStreamz support is added.
    elif isinstance(src, (int, float, complex, basestring, os.PathLike)):
        # If source is a file, return source_info where type=FILEPATH
        if os.path.isfile(src):
            return source_info(<string> str(src).encode())
        # If source expected to be a file, raise FileNotFoundError
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), src)
    else:
        raise TypeError("Unrecognized input type: {}".format(type(src)))
    return source_info(<char*>&buf[0], buf.shape[0])
