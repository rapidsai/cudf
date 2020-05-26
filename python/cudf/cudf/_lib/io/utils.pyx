# Copyright (c) 2020, NVIDIA CORPORATION.

from cpython.buffer cimport PyBUF_READ
from cpython.memoryview cimport PyMemoryView_FromMemory
from libcpp.memory cimport unique_ptr
from libcpp.pair cimport pair
from libcpp.string cimport string
from cudf._lib.cpp.io.types cimport source_info, sink_info, data_sink, io_type

import codecs
import errno
import io
import os

# Converts the Python source input to libcudf++ IO source_info
# with the appropriate type and source values
cdef source_info make_source_info(src) except*:
    cdef const unsigned char[::1] buf
    empty_buffer = False
    if isinstance(src, bytes):
        if (len(src) > 0):
            buf = src
        else:
            empty_buffer = True
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
    if empty_buffer is False:
        return source_info(<char*>&buf[0], buf.shape[0])
    else:
        return source_info(<char*>NULL, 0)


# Converts the Python sink input to libcudf++ IO sink_info.
cdef sink_info make_sink_info(src, unique_ptr[data_sink] * sink) except*:
    if isinstance(src, io.TextIOBase):
        # Files opened in text mode expect writes to be str rather than bytes,
        # which requires conversion from utf-8. If the underlying buffer is
        # utf-8, we can bypass this conversion by writing directly to it.
        if codecs.lookup(src.encoding).name not in {"utf-8", "ascii"}:
            raise NotImplementedError(f"Unsupported encoding {src.encoding}")
        sink.reset(new iobase_data_sink(src.buffer))
        return sink_info(sink.get())
    elif isinstance(src, io.IOBase):
        sink.reset(new iobase_data_sink(src))
        return sink_info(sink.get())
    elif isinstance(src, (basestring, os.PathLike)):
        return sink_info(<string> os.path.expanduser(src).encode())
    else:
        raise TypeError("Unrecognized input type: {}".format(type(src)))


# Adapts a python io.IOBase object as a libcudf++ IO data_sink. This lets you
# write from cudf to any python file-like object (File/BytesIO/SocketIO etc)
cdef cppclass iobase_data_sink(data_sink):
    object buf

    iobase_data_sink(object buf_):
        this.buf = buf_

    void host_write(const void * data, size_t size) with gil:
        buf.write(PyMemoryView_FromMemory(<char*>data, size, PyBUF_READ))

    void flush() with gil:
        buf.flush()

    size_t bytes_written() with gil:
        return buf.tell()
