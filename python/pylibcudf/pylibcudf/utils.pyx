# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from cython.operator import dereference

from libcpp.functional cimport reference_wrapper
from libcpp.vector cimport vector

from pylibcudf.libcudf.scalar.scalar cimport scalar

from .scalar cimport Scalar

from rmm.pylibrmm.stream cimport Stream
from rmm.pylibrmm.memory_resource cimport (
    DeviceMemoryResource,
    get_current_device_resource,
)

from rmm.pylibrmm.stream import DEFAULT_STREAM, PER_THREAD_DEFAULT_STREAM

from cuda.bindings import runtime

import os

# Check the environment for the variable CUDF_PER_THREAD_STREAM. If it is set,
# then set the module-scope CUDF_DEFAULT_STREAM variable here to
# rmm.pylibrmm.stream.PER_THREAD_DEFAULT_STREAM. Otherwise, it will default to
# rmm.pylibrmm.stream.DEFAULT_STREAM.
if os.getenv("CUDF_PER_THREAD_STREAM", "0") == "1":
    CUDF_DEFAULT_STREAM = PER_THREAD_DEFAULT_STREAM
else:
    CUDF_DEFAULT_STREAM = DEFAULT_STREAM

# This is a workaround for
# https://github.com/cython/cython/issues/4180
# when creating reference_wrapper[constscalar] in the constructor
ctypedef const scalar constscalar

cdef vector[reference_wrapper[const scalar]] _as_vector(list source):
    """Make a vector of reference_wrapper[const scalar] from a list of scalars."""
    cdef vector[reference_wrapper[const scalar]] c_scalars
    c_scalars.reserve(len(source))
    cdef Scalar slr
    for slr in source:
        c_scalars.push_back(
            reference_wrapper[constscalar](dereference((<Scalar?>slr).c_obj)))
    return c_scalars


def _is_concurrent_managed_access_supported():
    """Check the availability of concurrent managed access (UVM).

    Note that WSL2 does not support managed memory.
    """

    # Ensure CUDA is initialized before checking cudaDevAttrConcurrentManagedAccess
    runtime.cudaFree(0)

    device_id = 0
    err, supports_managed_access = runtime.cudaDeviceGetAttribute(
        runtime.cudaDeviceAttr.cudaDevAttrConcurrentManagedAccess, device_id
    )
    if err != runtime.cudaError_t.cudaSuccess:
        raise RuntimeError(
            f"Failed to check cudaDevAttrConcurrentManagedAccess with error {err}"
        )
    return supports_managed_access != 0


cdef Stream _get_stream(Stream stream = None):
    if stream is None:
        return CUDF_DEFAULT_STREAM
    return stream


cdef DeviceMemoryResource _get_memory_resource(DeviceMemoryResource mr = None):
    if mr is None:
        return get_current_device_resource()
    return mr
