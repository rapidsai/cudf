# Copyright (c) 2020, NVIDIA CORPORATION.

from cpython.buffer cimport PyBUF_READ
from cpython.memoryview cimport PyMemoryView_FromMemory
from libcpp.map cimport map
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp.string cimport string
from cudf._lib.cpp.io.types cimport source_info, io_type, host_buffer
from cudf._lib.cpp.io.types cimport sink_info, data_sink, datasource
from cudf._lib.io.datasource cimport Datasource

import codecs
import errno
import io
import os
import cudf

# Converts the Python source input to libcudf++ IO source_info
# with the appropriate type and source values
cdef source_info make_source_info(list src) except*:
    if not src:
        raise ValueError("Need to pass at least one source")

    cdef const unsigned char[::1] c_buffer
    cdef vector[host_buffer] c_host_buffers
    cdef vector[string] c_files
    cdef Datasource csrc
    empty_buffer = False
    if isinstance(src[0], bytes):
        empty_buffer = True
        for buffer in src:
            if (len(buffer) > 0):
                c_buffer = buffer
                c_host_buffers.push_back(host_buffer(<char*>&c_buffer[0],
                                                     c_buffer.shape[0]))
                empty_buffer = False
    elif isinstance(src[0], io.BytesIO):
        for bio in src:
            c_buffer = bio.getbuffer()  # check if empty?
            c_host_buffers.push_back(host_buffer(<char*>&c_buffer[0],
                                                 c_buffer.shape[0]))
    # Otherwise src is expected to be a numeric fd, string path, or PathLike.
    # TODO (ptaylor): Might need to update this check if accepted input types
    #                 change when UCX and/or cuStreamz support is added.
    elif isinstance(src[0], Datasource):
        csrc = src[0]
        return source_info(csrc.get_datasource())
    elif isinstance(src[0], (int, float, complex, basestring, os.PathLike)):
        # If source is a file, return source_info where type=FILEPATH
        if not all(os.path.isfile(file) for file in src):
            raise FileNotFoundError(errno.ENOENT,
                                    os.strerror(errno.ENOENT),
                                    src)

        files = [<string> str(elem).encode() for elem in src]
        c_files = files
        return source_info(c_files)
    else:
        raise TypeError("Unrecognized input type: {}".format(type(src[0])))

    if empty_buffer is True:
        c_host_buffers.push_back(host_buffer(<char*>NULL, 0))

    return source_info(c_host_buffers)

# Converts the Python sink input to libcudf++ IO sink_info.
cdef sink_info make_sink_info(src, unique_ptr[data_sink] & sink) except*:
    if isinstance(src, io.StringIO):
        sink.reset(new iobase_data_sink(src))
        return sink_info(sink.get())
    elif isinstance(src, io.TextIOBase):
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
        if isinstance(buf, io.StringIO):
            buf.write(PyMemoryView_FromMemory(<char*>data, size, PyBUF_READ)
                      .tobytes().decode())
        else:
            buf.write(PyMemoryView_FromMemory(<char*>data, size, PyBUF_READ))

    void flush() with gil:
        buf.flush()

    size_t bytes_written() with gil:
        return buf.tell()
