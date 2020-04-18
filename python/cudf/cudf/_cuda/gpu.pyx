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
from libc.stdlib cimport malloc
from cudf._cuda.gpu cimport underlying_type_attribute as c_attr


class CUDARuntimeError(RuntimeError):

    def __init__(self, cudaError_t status):
        self.status = status
        cdef bytes name = cudaGetErrorName(status)
        cdef bytes msg = cudaGetErrorString(status)
        super(CUDARuntimeError, self).__init__(
            '%s: %s' % (name.decode(), msg.decode()))

    def __reduce__(self):
        return (type(self), (self.status,))


class CudaDeviceAttr(IntEnum):
    cudaDevAttrMaxThreadsPerBlock = \
        <c_attr> cudaDeviceAttr.cudaDevAttrMaxThreadsPerBlock
    cudaDevAttrMaxBlockDimX = \
        <c_attr> cudaDeviceAttr.cudaDevAttrMaxBlockDimX
    cudaDevAttrMaxBlockDimY = \
        <c_attr> cudaDeviceAttr.cudaDevAttrMaxBlockDimY
    cudaDevAttrMaxBlockDimZ = \
        <c_attr> cudaDeviceAttr.cudaDevAttrMaxBlockDimZ
    cudaDevAttrMaxGridDimX = \
        <c_attr> cudaDeviceAttr.cudaDevAttrMaxGridDimX
    cudaDevAttrMaxGridDimY = \
        <c_attr> cudaDeviceAttr.cudaDevAttrMaxGridDimY
    cudaDevAttrMaxGridDimZ = \
        <c_attr> cudaDeviceAttr.cudaDevAttrMaxGridDimZ
    cudaDevAttrMaxSharedMemoryPerBlock = \
        <c_attr> cudaDeviceAttr.cudaDevAttrMaxSharedMemoryPerBlock
    cudaDevAttrTotalConstantMemory = \
        <c_attr> cudaDeviceAttr.cudaDevAttrTotalConstantMemory
    cudaDevAttrWarpSize = \
        <c_attr> cudaDeviceAttr.cudaDevAttrWarpSize
    cudaDevAttrMaxPitch = \
        <c_attr> cudaDeviceAttr.cudaDevAttrMaxPitch
    cudaDevAttrMaxRegistersPerBlock = \
        <c_attr> cudaDeviceAttr.cudaDevAttrMaxRegistersPerBlock
    cudaDevAttrClockRate = \
        <c_attr> cudaDeviceAttr.cudaDevAttrClockRate
    cudaDevAttrTextureAlignment = \
        <c_attr> cudaDeviceAttr.cudaDevAttrTextureAlignment
    cudaDevAttrGpuOverlap = \
        <c_attr> cudaDeviceAttr.cudaDevAttrGpuOverlap
    cudaDevAttrMultiProcessorCount = \
        <c_attr> cudaDeviceAttr.cudaDevAttrMultiProcessorCount
    cudaDevAttrKernelExecTimeout = \
        <c_attr> cudaDeviceAttr.cudaDevAttrKernelExecTimeout
    cudaDevAttrIntegrated = \
        <c_attr> cudaDeviceAttr.cudaDevAttrIntegrated
    cudaDevAttrCanMapHostMemory = \
        <c_attr> cudaDeviceAttr.cudaDevAttrCanMapHostMemory
    cudaDevAttrComputeMode = \
        <c_attr> cudaDeviceAttr.cudaDevAttrComputeMode
    cudaDevAttrMaxTexture1DWidth = \
        <c_attr> cudaDeviceAttr.cudaDevAttrMaxTexture1DWidth
    cudaDevAttrMaxTexture2DWidth = \
        <c_attr> cudaDeviceAttr.cudaDevAttrMaxTexture2DWidth
    cudaDevAttrMaxTexture2DHeight = \
        <c_attr> cudaDeviceAttr.cudaDevAttrMaxTexture2DHeight
    cudaDevAttrMaxTexture3DWidth = \
        <c_attr> cudaDeviceAttr.cudaDevAttrMaxTexture3DWidth
    cudaDevAttrMaxTexture3DHeight = \
        <c_attr> cudaDeviceAttr.cudaDevAttrMaxTexture3DHeight
    cudaDevAttrMaxTexture3DDepth = \
        <c_attr> cudaDeviceAttr.cudaDevAttrMaxTexture3DDepth
    cudaDevAttrMaxTexture2DLayeredWidth = \
        <c_attr> cudaDeviceAttr.cudaDevAttrMaxTexture2DLayeredWidth
    cudaDevAttrMaxTexture2DLayeredHeight = \
        <c_attr> cudaDeviceAttr.cudaDevAttrMaxTexture2DLayeredHeight
    cudaDevAttrMaxTexture2DLayeredLayers = \
        <c_attr> cudaDeviceAttr.cudaDevAttrMaxTexture2DLayeredLayers
    cudaDevAttrSurfaceAlignment = \
        <c_attr> cudaDeviceAttr.cudaDevAttrSurfaceAlignment
    cudaDevAttrConcurrentKernels = \
        <c_attr> cudaDeviceAttr.cudaDevAttrConcurrentKernels
    cudaDevAttrEccEnabled = \
        <c_attr> cudaDeviceAttr.cudaDevAttrEccEnabled
    cudaDevAttrPciBusId = \
        <c_attr> cudaDeviceAttr.cudaDevAttrPciBusId
    cudaDevAttrPciDeviceId = \
        <c_attr> cudaDeviceAttr.cudaDevAttrPciDeviceId
    cudaDevAttrTccDriver = \
        <c_attr> cudaDeviceAttr.cudaDevAttrTccDriver
    cudaDevAttrMemoryClockRate = \
        <c_attr> cudaDeviceAttr.cudaDevAttrMemoryClockRate
    cudaDevAttrGlobalMemoryBusWidth = \
        <c_attr> cudaDeviceAttr.cudaDevAttrGlobalMemoryBusWidth
    cudaDevAttrL2CacheSize = \
        <c_attr> cudaDeviceAttr.cudaDevAttrL2CacheSize
    cudaDevAttrMaxThreadsPerMultiProcessor = \
        <c_attr> cudaDeviceAttr.cudaDevAttrMaxThreadsPerMultiProcessor
    cudaDevAttrAsyncEngineCount = \
        <c_attr> cudaDeviceAttr.cudaDevAttrAsyncEngineCount
    cudaDevAttrUnifiedAddressing = \
        <c_attr> cudaDeviceAttr.cudaDevAttrUnifiedAddressing
    cudaDevAttrMaxTexture1DLayeredWidth = \
        <c_attr> cudaDeviceAttr.cudaDevAttrMaxTexture1DLayeredWidth
    cudaDevAttrMaxTexture1DLayeredLayers = \
        <c_attr> cudaDeviceAttr.cudaDevAttrMaxTexture1DLayeredLayers
    cudaDevAttrMaxTexture2DGatherWidth = \
        <c_attr> cudaDeviceAttr.cudaDevAttrMaxTexture2DGatherWidth
    cudaDevAttrMaxTexture2DGatherHeight = \
        <c_attr> cudaDeviceAttr.cudaDevAttrMaxTexture2DGatherHeight
    cudaDevAttrMaxTexture3DWidthAlt = \
        <c_attr> cudaDeviceAttr.cudaDevAttrMaxTexture3DWidthAlt
    cudaDevAttrMaxTexture3DHeightAlt = \
        <c_attr> cudaDeviceAttr.cudaDevAttrMaxTexture3DHeightAlt
    cudaDevAttrMaxTexture3DDepthAlt = \
        <c_attr> cudaDeviceAttr.cudaDevAttrMaxTexture3DDepthAlt
    cudaDevAttrPciDomainId = \
        <c_attr> cudaDeviceAttr.cudaDevAttrPciDomainId
    cudaDevAttrTexturePitchAlignment = \
        <c_attr> cudaDeviceAttr.cudaDevAttrTexturePitchAlignment
    cudaDevAttrMaxTextureCubemapWidth = \
        <c_attr> cudaDeviceAttr.cudaDevAttrMaxTextureCubemapWidth
    cudaDevAttrMaxTextureCubemapLayeredWidth = \
        <c_attr> cudaDeviceAttr.cudaDevAttrMaxTextureCubemapLayeredWidth
    cudaDevAttrMaxTextureCubemapLayeredLayers = \
        <c_attr> cudaDeviceAttr.cudaDevAttrMaxTextureCubemapLayeredLayers
    cudaDevAttrMaxSurface1DWidth = \
        <c_attr> cudaDeviceAttr.cudaDevAttrMaxSurface1DWidth
    cudaDevAttrMaxSurface2DWidth = \
        <c_attr> cudaDeviceAttr.cudaDevAttrMaxSurface2DWidth
    cudaDevAttrMaxSurface2DHeight = \
        <c_attr> cudaDeviceAttr.cudaDevAttrMaxSurface2DHeight
    cudaDevAttrMaxSurface3DWidth = \
        <c_attr> cudaDeviceAttr.cudaDevAttrMaxSurface3DWidth
    cudaDevAttrMaxSurface3DHeight = \
        <c_attr> cudaDeviceAttr.cudaDevAttrMaxSurface3DHeight
    cudaDevAttrMaxSurface3DDepth = \
        <c_attr> cudaDeviceAttr.cudaDevAttrMaxSurface3DDepth
    cudaDevAttrMaxSurface1DLayeredWidth = \
        <c_attr> cudaDeviceAttr.cudaDevAttrMaxSurface1DLayeredWidth
    cudaDevAttrMaxSurface1DLayeredLayers = \
        <c_attr> cudaDeviceAttr.cudaDevAttrMaxSurface1DLayeredLayers
    cudaDevAttrMaxSurface2DLayeredWidth = \
        <c_attr> cudaDeviceAttr.cudaDevAttrMaxSurface2DLayeredWidth
    cudaDevAttrMaxSurface2DLayeredHeight = \
        <c_attr> cudaDeviceAttr.cudaDevAttrMaxSurface2DLayeredHeight
    cudaDevAttrMaxSurface2DLayeredLayers = \
        <c_attr> cudaDeviceAttr.cudaDevAttrMaxSurface2DLayeredLayers
    cudaDevAttrMaxSurfaceCubemapWidth = \
        <c_attr> cudaDeviceAttr.cudaDevAttrMaxSurfaceCubemapWidth
    cudaDevAttrMaxSurfaceCubemapLayeredWidth = \
        <c_attr> cudaDeviceAttr.cudaDevAttrMaxSurfaceCubemapLayeredWidth
    cudaDevAttrMaxSurfaceCubemapLayeredLayers = \
        <c_attr> cudaDeviceAttr.cudaDevAttrMaxSurfaceCubemapLayeredLayers
    cudaDevAttrMaxTexture1DLinearWidth = \
        <c_attr> cudaDeviceAttr.cudaDevAttrMaxTexture1DLinearWidth
    cudaDevAttrMaxTexture2DLinearWidth = \
        <c_attr> cudaDeviceAttr.cudaDevAttrMaxTexture2DLinearWidth
    cudaDevAttrMaxTexture2DLinearHeight = \
        <c_attr> cudaDeviceAttr.cudaDevAttrMaxTexture2DLinearHeight
    cudaDevAttrMaxTexture2DLinearPitch = \
        <c_attr> cudaDeviceAttr.cudaDevAttrMaxTexture2DLinearPitch
    cudaDevAttrMaxTexture2DMipmappedWidth = \
        <c_attr> cudaDeviceAttr.cudaDevAttrMaxTexture2DMipmappedWidth
    cudaDevAttrMaxTexture2DMipmappedHeight = \
        <c_attr> cudaDeviceAttr.cudaDevAttrMaxTexture2DMipmappedHeight
    cudaDevAttrComputeCapabilityMajor = \
        <c_attr> cudaDeviceAttr.cudaDevAttrComputeCapabilityMajor
    cudaDevAttrComputeCapabilityMinor = \
        <c_attr> cudaDeviceAttr.cudaDevAttrComputeCapabilityMinor
    cudaDevAttrMaxTexture1DMipmappedWidth = \
        <c_attr> cudaDeviceAttr.cudaDevAttrMaxTexture1DMipmappedWidth
    cudaDevAttrStreamPrioritiesSupported = \
        <c_attr> cudaDeviceAttr.cudaDevAttrStreamPrioritiesSupported
    cudaDevAttrGlobalL1CacheSupported = \
        <c_attr> cudaDeviceAttr.cudaDevAttrGlobalL1CacheSupported
    cudaDevAttrLocalL1CacheSupported = \
        <c_attr> cudaDeviceAttr.cudaDevAttrLocalL1CacheSupported
    cudaDevAttrMaxSharedMemoryPerMultiprocessor = \
        <c_attr> cudaDeviceAttr.cudaDevAttrMaxSharedMemoryPerMultiprocessor
    cudaDevAttrMaxRegistersPerMultiprocessor = \
        <c_attr> cudaDeviceAttr.cudaDevAttrMaxRegistersPerMultiprocessor
    cudaDevAttrManagedMemory = \
        <c_attr> cudaDeviceAttr.cudaDevAttrManagedMemory
    cudaDevAttrIsMultiGpuBoard = \
        <c_attr> cudaDeviceAttr.cudaDevAttrIsMultiGpuBoard
    cudaDevAttrMultiGpuBoardGroupID = \
        <c_attr> cudaDeviceAttr.cudaDevAttrMultiGpuBoardGroupID
    cudaDevAttrHostNativeAtomicSupported = \
        <c_attr> cudaDeviceAttr.cudaDevAttrHostNativeAtomicSupported
    cudaDevAttrSingleToDoublePrecisionPerfRatio = \
        <c_attr> cudaDeviceAttr.cudaDevAttrSingleToDoublePrecisionPerfRatio
    cudaDevAttrPageableMemoryAccess = \
        <c_attr> cudaDeviceAttr.cudaDevAttrPageableMemoryAccess
    cudaDevAttrConcurrentManagedAccess = \
        <c_attr> cudaDeviceAttr.cudaDevAttrConcurrentManagedAccess
    cudaDevAttrComputePreemptionSupported = \
        <c_attr> cudaDeviceAttr.cudaDevAttrComputePreemptionSupported
    cudaDevAttrCanUseHostPointerForRegisteredMem = \
        <c_attr> cudaDeviceAttr.cudaDevAttrCanUseHostPointerForRegisteredMem
    cudaDevAttrReserved92 = \
        <c_attr> cudaDeviceAttr.cudaDevAttrReserved92
    cudaDevAttrReserved93 = \
        <c_attr> cudaDeviceAttr.cudaDevAttrReserved93
    cudaDevAttrReserved94 = \
        <c_attr> cudaDeviceAttr.cudaDevAttrReserved94
    cudaDevAttrCooperativeLaunch = \
        <c_attr> cudaDeviceAttr.cudaDevAttrCooperativeLaunch
    cudaDevAttrCooperativeMultiDeviceLaunch = \
        <c_attr> cudaDeviceAttr.cudaDevAttrCooperativeMultiDeviceLaunch
    cudaDevAttrMaxSharedMemoryPerBlockOptin = \
        <c_attr> cudaDeviceAttr.cudaDevAttrMaxSharedMemoryPerBlockOptin
    cudaDevAttrCanFlushRemoteWrites = \
        <c_attr> cudaDeviceAttr.cudaDevAttrCanFlushRemoteWrites
    cudaDevAttrHostRegisterSupported = \
        <c_attr> cudaDeviceAttr.cudaDevAttrHostRegisterSupported
    cudaDevAttrPageableMemoryAccessUsesHostPageTables = \
        <c_attr> \
        cudaDeviceAttr.cudaDevAttrPageableMemoryAccessUsesHostPageTables
    cudaDevAttrDirectManagedMemAccessFromHost = \
        <c_attr> cudaDeviceAttr.cudaDevAttrDirectManagedMemAccessFromHost


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
    status = cudaDriverGetVersion(&version)
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
    status = cudaRuntimeGetVersion(&version)
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
    status = cudaGetDeviceCount(&count)
    if status != 0:
        raise CUDARuntimeError(status)
    return count


def getDeviceAttribute(object attr, int device):
    """
    Returns information about the device.

    Parameters
    ----------
        attr : object (CudaDeviceAttr)
            Device attribute to query
        device : int
            Device number to query

    This function automatically raises CUDARuntimeError with error message
    and status code.
    """

    cdef int value
    status = cudaDeviceGetAttribute(&value, attr, device)
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
    status = cudaGetDeviceProperties(&prop, device)
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

    This function automatically raises CUDARuntimeError with error message
    and status code.
    """

    cdef char* device_name = <char*> malloc(256 * sizeof(char))
    status = cuDeviceGetName(device_name, 256, device)
    if status != 0:
        raise CUDARuntimeError(status)
    return device_name
