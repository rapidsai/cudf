# Copyright (c) 2020, NVIDIA CORPORATION.


cdef extern from "cudf/utilities/device.hpp" namespace \
        "cudf::experimental" nogil:

    cdef int get_cuda_runtime_version() except +
    cdef int get_gpu_device_count() except +
    cdef int get_cuda_latest_supported_driver_version() except +
