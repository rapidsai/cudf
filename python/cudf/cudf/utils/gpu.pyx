# Copyright (c) 2020, NVIDIA CORPORATION.

from cudf.utils.gpu cimport (
    get_cuda_runtime_version as cpp_get_cuda_runtime_version,
    get_gpu_device_count as cpp_get_gpu_device_count,
    get_cuda_latest_supported_driver_version as
    cpp_get_cuda_latest_supported_driver_version
)


def get_cuda_runtime_version():
    cdef int c_result
    with nogil:
        c_result = cpp_get_cuda_runtime_version()
    return c_result


def get_gpu_device_count():
    cdef int c_result
    with nogil:
        c_result = cpp_get_gpu_device_count()
    return c_result


def get_cuda_latest_supported_driver_version():
    cdef int c_result
    with nogil:
        c_result = cpp_get_cuda_latest_supported_driver_version()
    return c_result
