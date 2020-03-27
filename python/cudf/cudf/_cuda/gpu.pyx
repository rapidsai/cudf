# Copyright (c) 2020, NVIDIA CORPORATION.

from cudf._cuda.gpu cimport (
    cudaDriverGetVersion,
    cudaRuntimeGetVersion,
    cudaGetDeviceCount,
    cudaDeviceGetAttribute,
    cudaDeviceAttr
)
from enum import IntEnum
from cudf._cuda.gpu cimport underlying_type_attribute as c_attr


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

    This function returns -1 if driver version is NULL.
    """
    cdef int version
    status = cudaDriverGetVersion(&version)
    return -1 if status != 0 else version


def runtimeGetVersion():
    """
    Returns the version number of the current CUDA Runtime instance.
    The version is returned as (1000 major + 10 minor). For example,
    CUDA 9.2 would be represented by 9020.

    This function returns -1 if runtime version is NULL.
    """

    cdef int version
    status = cudaRuntimeGetVersion(&version)
    return -1 if status != 0 else version


def getDeviceCount():
    """
    Returns the number of devices with compute capability greater or
    equal to 2.0 that are available for execution.

    This function returns -1 if NULL device pointer is assigned.
    """

    cdef int count
    status = cudaGetDeviceCount(&count)
    return -1 if status != 0 else count


def getDeviceAttribute(attr, device):
    """
    Returns information about the device.

    Parameters
        attr
            Device attribute to query
        device
            Device number to query
    """

    cdef int value
    status = cudaDeviceGetAttribute(&value, attr, device)
    return -1 if status != 0 else value
