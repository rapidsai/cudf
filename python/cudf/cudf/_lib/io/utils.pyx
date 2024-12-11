# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from cpython.buffer cimport PyBUF_READ
from cpython.memoryview cimport PyMemoryView_FromMemory
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libcpp.utility cimport move
from libcpp.vector cimport vector

from pylibcudf.libcudf.io.data_sink cimport data_sink
from pylibcudf.libcudf.io.types cimport (
    column_name_info,
    sink_info,
)

from cudf._lib.column cimport Column

import codecs
import io
import os

from cudf.core.dtypes import StructDtype

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


cdef add_df_col_struct_names(df, child_names_dict):
    for name, child_names in child_names_dict.items():
        col = df._data[name]

        df._data[name] = update_col_struct_field_names(col, child_names)


cdef update_col_struct_field_names(Column col, child_names):
    if col.children:
        children = list(col.children)
        for i, (child, names) in enumerate(zip(children, child_names.values())):
            children[i] = update_col_struct_field_names(
                child,
                names
            )
        col.set_base_children(tuple(children))

    if isinstance(col.dtype, StructDtype):
        col = col._rename_fields(
            child_names.keys()
        )

    return col


cdef update_struct_field_names(
    table,
    vector[column_name_info]& schema_info
):
    # Deprecated, remove in favor of add_col_struct_names
    # when a reader is ported to pylibcudf
    for i, (name, col) in enumerate(table._column_labels_and_values):
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
