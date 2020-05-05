# Copyright (c) 2020, NVIDIA CORPORATION.

from cpython.buffer cimport PyBUF_READ
from cpython.memoryview cimport PyMemoryView_FromMemory
from libcpp.memory cimport unique_ptr
from libcpp.pair cimport pair
from libcpp.string cimport string
from libcpp.vector cimport vector
from cudf._lib.cpp.io.types cimport source_info, sink_info, data_sink, io_type

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
    elif isinstance(src, HostBuffer):
        return source_info((<HostBuffer>src).buf.data(),
                           (<HostBuffer>src).buf.size())
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
cdef sink_info make_sink_info(src, unique_ptr[data_sink] * data) except*:
    if isinstance(src, HostBuffer):
        return sink_info(&(<HostBuffer>src).buf)
    if isinstance(src, io.IOBase):
        data.reset(new iobase_data_sink(src))
        return sink_info(data.get())
    elif isinstance(src, (int, float, complex, basestring, os.PathLike)):
        return sink_info(<string> str(src).encode())
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


cdef class HostBuffer:
    """ HostBuffer lets you spill cudf DataFrames from device memory to host
    memory. Once in host memory the dataframe can either be re-loaded back
    into gpu memory, or spilled to disk. This is designed to reduce the amount
    of unnecessary host memory copies.

    Examples
    --------
    .. code-block:: python
      import shutil
      import cudf

      # read cudf DataFrame into buffer on host
      df = cudf.DataFrame({'a': [1, 1, 2], 'b': [1, 2, 3]})
      buffer = cudf.io.HostBuffer()
      df.to_parquet(buffer)

      # Copy HostBuffer back to DataFrame
      cudf.read_parquet(df)

      # Write HostBuffer to disk
      shutil.copyfileobj(buffer, open("output.parquet", "wb"))
    """
    cdef vector[char] buf
    cdef size_t pos

    def __cinit__(self, int initial_capacity=0):
        self.pos = 0
        if initial_capacity:
            self.buf.reserve(initial_capacity)

    def __len__(self):
        return self.buf.size()

    def seek(self, size_t pos):
        self.pos = pos

    def tell(self):
        return self.pos

    def readall(self):
        return self.read(-1)

    def read(self, int n=-1):
        if self.pos >= self.buf.size():
            return b""

        cdef size_t count = n
        if ((n < 0) or (n > self.buf.size() - self.pos)):
            count = self.buf.size() - self.pos

        cdef size_t start = self.pos
        self.pos += count

        return PyMemoryView_FromMemory(self.buf.data() + start, count,
                                       PyBUF_READ)
