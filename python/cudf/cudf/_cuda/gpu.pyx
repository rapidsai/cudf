# Copyright (c) 2020, NVIDIA CORPORATION.

from cudf._cuda.gpu cimport (
    cudaDriverGetVersion,
    cudaRuntimeGetVersion,
    cudaGetDeviceCount,
    cudaDeviceGetAttribute,
    cudaDeviceAttr,
    cudaGetDeviceProperties,
    cudaDeviceProp,
    cudaGetErrorName,
    cudaGetErrorString,
    cudaError_t,
    CUresult,
    cuDeviceGetName,
    cuGetErrorName,
    cuGetErrorString
)
from enum import IntEnum


class CUDARuntimeError(RuntimeError):

    def __init__(self, cudaError_t status):
        self.status = status
        cdef str name = cudaGetErrorName(status).decode()
        cdef str msg = cudaGetErrorString(status).decode()
        super(CUDARuntimeError, self).__init__(
            '%s: %s' % (name, msg))

    def __reduce__(self):
        return (type(self), (self.status,))


class CUDADriverError(RuntimeError):

    def __init__(self, CUresult status):
        self.status = status

        cdef const char* name_cstr
        cdef CUresult name_status = cuGetErrorName(status, &name_cstr)
        if name_status != 0:
            raise CUDADriverError(name_status)

        cdef const char* msg_cstr
        cdef CUresult msg_status = cuGetErrorString(status, &msg_cstr)
        if msg_status != 0:
            raise CUDADriverError(msg_status)

        cdef str name = name_cstr.decode()
        cdef str msg = msg_cstr.decode()

        super(CUDADriverError, self).__init__(
            '%s: %s' % (name, msg))

    def __reduce__(self):
        return (type(self), (self.status,))


def driverGetVersion():
    """
    Returns in the latest version of CUDA supported by the driver.
    The version is returned as (1000 major + 10 minor). For example,
    CUDA 9.2 would be represented by 9020. If no driver is installed,
    then 0 is returned as the driver version.

    This function automatically raises CUDARuntimeError with error message
    and status code.
    """
    cdef int version
    cdef cudaError_t status = cudaDriverGetVersion(&version)
    if status != 0:
        raise CUDARuntimeError(status)
    return version


def runtimeGetVersion():
    """
    Returns the version number of the current CUDA Runtime instance.
    The version is returned as (1000 major + 10 minor). For example,
    CUDA 9.2 would be represented by 9020.

    This function automatically raises CUDARuntimeError with error message
    and status code.
    """

    cdef int version
    cdef cudaError_t status = cudaRuntimeGetVersion(&version)
    if status != 0:
        raise CUDARuntimeError(status)
    return version


def getDeviceCount():
    """
    Returns the number of devices with compute capability greater or
    equal to 2.0 that are available for execution.

    This function automatically raises CUDARuntimeError with error message
    and status code.
    """

    cdef int count
    cdef cudaError_t status = cudaGetDeviceCount(&count)

    if status != 0:
        raise CUDARuntimeError(status)
    return count


def getDeviceAttribute(cudaDeviceAttr attr, int device):
    """
    Returns information about the device.

    Parameters
    ----------
        attr : cudaDeviceAttr
            Device attribute to query
        device : int
            Device number to query

    This function automatically raises CUDARuntimeError with error message
    and status code.
    """

    cdef int value
    cdef cudaError_t status = cudaDeviceGetAttribute(&value, attr, device)
    if status != 0:
        raise CUDARuntimeError(status)
    return value


def getDeviceProperties(int device):
    """
    Returns information about the compute-device.

    Parameters
    ----------
        device : int
            Device number to query

    This function automatically raises CUDARuntimeError with error message
    and status code.
    """

    cdef cudaDeviceProp prop
    cdef cudaError_t status = cudaGetDeviceProperties(&prop, device)
    if status != 0:
        raise CUDARuntimeError(status)
    return prop


def deviceGetName(int device):
    """
    Returns an identifer string for the device.

    Parameters
    ----------
        device : int
            Device number to query

    This function automatically raises CUDADriverError with error message
    and status code.
    """

    cdef char[256] device_name
    cdef CUresult status = cuDeviceGetName(
        device_name,
        sizeof(device_name),
        device
    )
    if status != 0:
        raise CUDADriverError(status)
    return device_name.decode()
