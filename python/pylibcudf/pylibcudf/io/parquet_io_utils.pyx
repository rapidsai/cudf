# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
"""IO utilities for the Parquet."""

from libc.stddef cimport size_t
from libc.stdint cimport uintptr_t
from libcpp.memory cimport make_unique, unique_ptr
from libcpp.pair cimport pair
from libcpp.utility cimport move
from libcpp.vector cimport vector
from cython.operator cimport dereference

from rmm.librmm.device_buffer cimport device_buffer
from rmm.pylibrmm.device_buffer cimport DeviceBuffer
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource
from rmm.pylibrmm.stream cimport Stream

from pylibcudf.gpumemoryview cimport gpumemoryview
from pylibcudf.io.text cimport ByteRangeInfo
from pylibcudf.io.types cimport SourceInfo
from pylibcudf.libcudf.io.datasource cimport datasource, make_datasources
from pylibcudf.libcudf.io.parquet_io_utils cimport (
    const_byte_range_info,
    const_uint8_t,
    fetch_byte_ranges_to_device as cpp_fetch_byte_ranges_to_device,
)
from pylibcudf.libcudf.io.text cimport byte_range_info
from pylibcudf.libcudf.utilities.span cimport device_span, host_span
from pylibcudf.utils cimport _get_memory_resource, _get_stream

__all__ = ["fetch_byte_ranges_to_device"]


def fetch_byte_ranges_to_device(
    SourceInfo source_info,
    list byte_ranges,
    object stream=None,
    object mr=None,
) -> list[gpumemoryview]:
    """Fetch byte ranges from a Parquet source into device memory.

    Parameters
    ----------
    source_info : SourceInfo
        Source describing a single Parquet file.
    byte_ranges : list[ByteRangeInfo]
        Byte ranges to fetch, as returned by
        :meth:`~pylibcudf.io.experimental.HybridScanReader.filter_column_chunks_byte_ranges`,
        :meth:`~pylibcudf.io.experimental.HybridScanReader.payload_column_chunks_byte_ranges`,
        or
        :meth:`~pylibcudf.io.experimental.HybridScanReader.all_column_chunks_byte_ranges`.
    stream : Stream, optional
        CUDA stream.
    mr : DeviceMemoryResource, optional
        Device memory resource.

    Returns
    -------
    list[gpumemoryview]
        One view per byte range. Each view holds a reference to the
        :class:`~rmm.DeviceBuffer` that owns its memory, keeping the
        allocation alive for as long as the view is referenced.

    Raises
    ------
    ValueError
        If ``source_info`` does not describe exactly one source.
    """
    cdef Stream _stream = _get_stream(stream)
    cdef DeviceMemoryResource _mr = _get_memory_resource(mr)
    cdef vector[unique_ptr[datasource]] sources = make_datasources(source_info.c_obj)
    if sources.size() != 1:
        raise ValueError(
            f"fetch_byte_ranges_to_device requires exactly one source, "
            f"got {sources.size()}"
        )

    cdef vector[byte_range_info] ranges_vec
    cdef ByteRangeInfo bri
    for bri in byte_ranges:
        ranges_vec.push_back(bri.c_obj)

    cdef pair[vector[device_buffer], vector[device_span[const_uint8_t]]] fetched
    with nogil:
        fetched = cpp_fetch_byte_ranges_to_device(
            dereference(sources[0]),
            host_span[const_byte_range_info](ranges_vec.data(), ranges_vec.size()),
            _stream.view(),
            _mr.get_mr(),
        )

    # Wrap the device_buffer as a Python DeviceBuffer that owns the allocation.
    # All views share a reference to it, keeping the memory alive.
    cdef DeviceBuffer owner = DeviceBuffer.c_from_unique_ptr(
        make_unique[device_buffer](move(fetched.first[0])),
        _stream,
        _mr,
    )

    cdef gpumemoryview gmv
    cdef uintptr_t ptr
    cdef size_t n
    result = []
    for i in range(fetched.second.size()):
        ptr = <uintptr_t>fetched.second[i].data()
        n = fetched.second[i].size()
        gmv = gpumemoryview.__new__(gpumemoryview)
        gmv.ptr = ptr
        gmv.nbytes = n
        gmv.obj = owner
        gmv.cai = {
            "shape": (n,),
            "strides": None,
            "typestr": "|u1",
            "data": (ptr, False),
            "version": 3,
        }
        result.append(gmv)
    return result
