# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from cpython.buffer cimport PyBUF_READ
from cpython.memoryview cimport PyMemoryView_FromMemory
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libcpp.utility cimport move
from libcpp.vector cimport vector

from cudf._lib.column cimport Column
from cudf._lib.io.datasource cimport Datasource
from cudf._lib.pylibcudf.libcudf.io.data_sink cimport data_sink
from cudf._lib.pylibcudf.libcudf.io.datasource cimport datasource
from cudf._lib.pylibcudf.libcudf.io.types cimport (
    column_name_info,
    host_buffer,
    sink_info,
    source_info,
)

import codecs
import errno
import io
import os

from cudf.core.dtypes import StructDtype


# Converts the Python source input to libcudf IO source_info
# with the appropriate type and source values
cdef source_info make_source_info(list src) except*:
    if not src:
        raise ValueError("Need to pass at least one source")

    cdef const unsigned char[::1] c_buffer
    cdef vector[host_buffer] c_host_buffers
    cdef vector[string] c_files
    cdef Datasource csrc
    cdef vector[datasource*] c_datasources
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
        for csrc in src:
            c_datasources.push_back(csrc.get_datasource())
        return source_info(c_datasources)
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

# Converts the Python sink input to libcudf IO sink_info.
cdef sink_info make_sinks_info(
    list src, vector[unique_ptr[data_sink]] & sink
) except*:
    cdef vector[data_sink *] data_sinks
    cdef vector[string] paths
    if isinstance(src[0], io.StringIO):
        data_sinks.reserve(len(src))
        for s in src:
            sink.push_back(unique_ptr[data_sink](new iobase_data_sink(s)))
            data_sinks.push_back(sink.back().get())
        return sink_info(data_sinks)
    elif isinstance(src[0], io.TextIOBase):
        data_sinks.reserve(len(src))
        for s in src:
            # Files opened in text mode expect writes to be str rather than
            # bytes, which requires conversion from utf-8. If the underlying
            # buffer is utf-8, we can bypass this conversion by writing
            # directly to it.
            if codecs.lookup(s.encoding).name not in {"utf-8", "ascii"}:
                raise NotImplementedError(f"Unsupported encoding {s.encoding}")
            sink.push_back(
                unique_ptr[data_sink](new iobase_data_sink(s.buffer))
            )
            data_sinks.push_back(sink.back().get())
        return sink_info(data_sinks)
    elif isinstance(src[0], io.IOBase):
        data_sinks.reserve(len(src))
        for s in src:
            sink.push_back(unique_ptr[data_sink](new iobase_data_sink(s)))
            data_sinks.push_back(sink.back().get())
        return sink_info(data_sinks)
    elif isinstance(src[0], (basestring, os.PathLike)):
        paths.reserve(len(src))
        for s in src:
            paths.push_back(<string> os.path.expanduser(s).encode())
        return sink_info(move(paths))
    else:
        raise TypeError("Unrecognized input type: {}".format(type(src)))


cdef sink_info make_sink_info(src, unique_ptr[data_sink] & sink) except*:
    cdef vector[unique_ptr[data_sink]] datasinks
    cdef sink_info info = make_sinks_info([src], datasinks)
    if not datasinks.empty():
        sink.swap(datasinks[0])
    return info


# Adapts a python io.IOBase object as a libcudf IO data_sink. This lets you
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


cdef update_struct_field_names(
    table,
    vector[column_name_info]& schema_info
):
    for i, (name, col) in enumerate(table._data.items()):
        table._data[name] = update_column_struct_field_names(
            col, schema_info[i]
        )


cdef Column update_column_struct_field_names(
    Column col,
    column_name_info& info
):
    cdef vector[string] field_names

    if col.children:
        children = list(col.children)
        for i, child in enumerate(children):
            children[i] = update_column_struct_field_names(
                child,
                info.children[i]
            )
        col.set_base_children(tuple(children))

    if isinstance(col.dtype, StructDtype):
        field_names.reserve(len(col.base_children))
        for i in range(info.children.size()):
            field_names.push_back(info.children[i].name)
        col = col._rename_fields(
            field_names
        )

    return col
