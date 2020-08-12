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
from cudf._cuda.gpu cimport underlying_type_attribute as c_attr
from cudf._cuda.gpu cimport underlying_type_error


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


class CudaError(IntEnum):
    cudaSuccess = \
        <underlying_type_error> cudaError.cudaSuccess
    cudaErrorInvalidValue = \
        <underlying_type_error> cudaError.cudaErrorInvalidValue
    cudaErrorMemoryAllocation = \
        <underlying_type_error> cudaError.cudaErrorMemoryAllocation
    cudaErrorInitializationError = \
        <underlying_type_error> cudaError.cudaErrorInitializationError
    cudaErrorCudartUnloading = \
        <underlying_type_error> cudaError.cudaErrorCudartUnloading
    cudaErrorProfilerDisabled = \
        <underlying_type_error> cudaError.cudaErrorProfilerDisabled
    cudaErrorProfilerNotInitialized = \
        <underlying_type_error> cudaError.cudaErrorProfilerNotInitialized
    cudaErrorProfilerAlreadyStarted = \
        <underlying_type_error> cudaError.cudaErrorProfilerAlreadyStarted
    cudaErrorProfilerAlreadyStopped = \
        <underlying_type_error> cudaError.cudaErrorProfilerAlreadyStopped
    cudaErrorInvalidConfiguration = \
        <underlying_type_error> cudaError.cudaErrorInvalidConfiguration
    cudaErrorInvalidPitchValue = \
        <underlying_type_error> cudaError.cudaErrorInvalidPitchValue
    cudaErrorInvalidSymbol = \
        <underlying_type_error> cudaError.cudaErrorInvalidSymbol
    cudaErrorInvalidHostPointer = \
        <underlying_type_error> cudaError.cudaErrorInvalidHostPointer
    cudaErrorInvalidDevicePointer = \
        <underlying_type_error> cudaError.cudaErrorInvalidDevicePointer
    cudaErrorInvalidTexture = \
        <underlying_type_error> cudaError.cudaErrorInvalidTexture
    cudaErrorInvalidTextureBinding = \
        <underlying_type_error> cudaError.cudaErrorInvalidTextureBinding
    cudaErrorInvalidChannelDescriptor = \
        <underlying_type_error> cudaError.cudaErrorInvalidChannelDescriptor
    cudaErrorInvalidMemcpyDirection = \
        <underlying_type_error> cudaError.cudaErrorInvalidMemcpyDirection
    cudaErrorAddressOfConstant = \
        <underlying_type_error> cudaError.cudaErrorAddressOfConstant
    cudaErrorTextureFetchFailed = \
        <underlying_type_error> cudaError.cudaErrorTextureFetchFailed
    cudaErrorTextureNotBound = \
        <underlying_type_error> cudaError.cudaErrorTextureNotBound
    cudaErrorSynchronizationError = \
        <underlying_type_error> cudaError.cudaErrorSynchronizationError
    cudaErrorInvalidFilterSetting = \
        <underlying_type_error> cudaError.cudaErrorInvalidFilterSetting
    cudaErrorInvalidNormSetting = \
        <underlying_type_error> cudaError.cudaErrorInvalidNormSetting
    cudaErrorMixedDeviceExecution = \
        <underlying_type_error> cudaError.cudaErrorMixedDeviceExecution
    cudaErrorNotYetImplemented = \
        <underlying_type_error> cudaError.cudaErrorNotYetImplemented
    cudaErrorMemoryValueTooLarge = \
        <underlying_type_error> cudaError.cudaErrorMemoryValueTooLarge
    cudaErrorInsufficientDriver = \
        <underlying_type_error> cudaError.cudaErrorInsufficientDriver
    cudaErrorInvalidSurface = \
        <underlying_type_error> cudaError.cudaErrorInvalidSurface
    cudaErrorDuplicateVariableName = \
        <underlying_type_error> cudaError.cudaErrorDuplicateVariableName
    cudaErrorDuplicateTextureName = \
        <underlying_type_error> cudaError.cudaErrorDuplicateTextureName
    cudaErrorDuplicateSurfaceName = \
        <underlying_type_error> cudaError.cudaErrorDuplicateSurfaceName
    cudaErrorDevicesUnavailable = \
        <underlying_type_error> cudaError.cudaErrorDevicesUnavailable
    cudaErrorIncompatibleDriverContext = \
        <underlying_type_error> cudaError.cudaErrorIncompatibleDriverContext
    cudaErrorMissingConfiguration = \
        <underlying_type_error> cudaError.cudaErrorMissingConfiguration
    cudaErrorPriorLaunchFailure = \
        <underlying_type_error> cudaError.cudaErrorPriorLaunchFailure
    cudaErrorLaunchMaxDepthExceeded = \
        <underlying_type_error> cudaError.cudaErrorLaunchMaxDepthExceeded
    cudaErrorLaunchFileScopedTex = \
        <underlying_type_error> cudaError.cudaErrorLaunchFileScopedTex
    cudaErrorLaunchFileScopedSurf = \
        <underlying_type_error> cudaError.cudaErrorLaunchFileScopedSurf
    cudaErrorSyncDepthExceeded = \
        <underlying_type_error> cudaError.cudaErrorSyncDepthExceeded
    cudaErrorLaunchPendingCountExceeded = \
        <underlying_type_error> cudaError.cudaErrorLaunchPendingCountExceeded
    cudaErrorInvalidDeviceFunction = \
        <underlying_type_error> cudaError.cudaErrorInvalidDeviceFunction
    cudaErrorNoDevice = \
        <underlying_type_error> cudaError.cudaErrorNoDevice
    cudaErrorInvalidDevice = \
        <underlying_type_error> cudaError.cudaErrorInvalidDevice
    cudaErrorStartupFailure = \
        <underlying_type_error> cudaError.cudaErrorStartupFailure
    cudaErrorInvalidKernelImage = \
        <underlying_type_error> cudaError.cudaErrorInvalidKernelImage
    cudaErrorDeviceUninitialized = \
        <underlying_type_error> cudaError.cudaErrorDeviceUninitialized
    cudaErrorMapBufferObjectFailed = \
        <underlying_type_error> cudaError.cudaErrorMapBufferObjectFailed
    cudaErrorUnmapBufferObjectFailed = \
        <underlying_type_error> cudaError.cudaErrorUnmapBufferObjectFailed
    cudaErrorArrayIsMapped = \
        <underlying_type_error> cudaError.cudaErrorArrayIsMapped
    cudaErrorAlreadyMapped = \
        <underlying_type_error> cudaError.cudaErrorAlreadyMapped
    cudaErrorNoKernelImageForDevice = \
        <underlying_type_error> cudaError.cudaErrorNoKernelImageForDevice
    cudaErrorAlreadyAcquired = \
        <underlying_type_error> cudaError.cudaErrorAlreadyAcquired
    cudaErrorNotMapped = \
        <underlying_type_error> cudaError.cudaErrorNotMapped
    cudaErrorNotMappedAsArray = \
        <underlying_type_error> cudaError.cudaErrorNotMappedAsArray
    cudaErrorNotMappedAsPointer = \
        <underlying_type_error> cudaError.cudaErrorNotMappedAsPointer
    cudaErrorECCUncorrectable = \
        <underlying_type_error> cudaError.cudaErrorECCUncorrectable
    cudaErrorUnsupportedLimit = \
        <underlying_type_error> cudaError.cudaErrorUnsupportedLimit
    cudaErrorDeviceAlreadyInUse = \
        <underlying_type_error> cudaError.cudaErrorDeviceAlreadyInUse
    cudaErrorPeerAccessUnsupported = \
        <underlying_type_error> cudaError.cudaErrorPeerAccessUnsupported
    cudaErrorInvalidPtx = \
        <underlying_type_error> cudaError.cudaErrorInvalidPtx
    cudaErrorInvalidGraphicsContext = \
        <underlying_type_error> cudaError.cudaErrorInvalidGraphicsContext
    cudaErrorNvlinkUncorrectable = \
        <underlying_type_error> cudaError.cudaErrorNvlinkUncorrectable
    cudaErrorJitCompilerNotFound = \
        <underlying_type_error> cudaError.cudaErrorJitCompilerNotFound
    cudaErrorInvalidSource = \
        <underlying_type_error> cudaError.cudaErrorInvalidSource
    cudaErrorFileNotFound = \
        <underlying_type_error> cudaError.cudaErrorFileNotFound
    cudaErrorSharedObjectSymbolNotFound = \
        <underlying_type_error> cudaError.cudaErrorSharedObjectSymbolNotFound
    cudaErrorSharedObjectInitFailed = \
        <underlying_type_error> cudaError.cudaErrorSharedObjectInitFailed
    cudaErrorOperatingSystem = \
        <underlying_type_error> cudaError.cudaErrorOperatingSystem
    cudaErrorInvalidResourceHandle = \
        <underlying_type_error> cudaError.cudaErrorInvalidResourceHandle
    cudaErrorIllegalState = \
        <underlying_type_error> cudaError.cudaErrorIllegalState
    cudaErrorSymbolNotFound = \
        <underlying_type_error> cudaError.cudaErrorSymbolNotFound
    cudaErrorNotReady = \
        <underlying_type_error> cudaError.cudaErrorNotReady
    cudaErrorIllegalAddress = \
        <underlying_type_error> cudaError.cudaErrorIllegalAddress
    cudaErrorLaunchOutOfResources = \
        <underlying_type_error> cudaError.cudaErrorLaunchOutOfResources
    cudaErrorLaunchTimeout = \
        <underlying_type_error> cudaError.cudaErrorLaunchTimeout
    cudaErrorLaunchIncompatibleTexturing = \
        <underlying_type_error> cudaError.cudaErrorLaunchIncompatibleTexturing
    cudaErrorPeerAccessAlreadyEnabled = \
        <underlying_type_error> cudaError.cudaErrorPeerAccessAlreadyEnabled
    cudaErrorPeerAccessNotEnabled = \
        <underlying_type_error> cudaError.cudaErrorPeerAccessNotEnabled
    cudaErrorSetOnActiveProcess = \
        <underlying_type_error> cudaError.cudaErrorSetOnActiveProcess
    cudaErrorContextIsDestroyed = \
        <underlying_type_error> cudaError.cudaErrorContextIsDestroyed
    cudaErrorAssert = \
        <underlying_type_error> cudaError.cudaErrorAssert
    cudaErrorTooManyPeers = \
        <underlying_type_error> cudaError.cudaErrorTooManyPeers
    cudaErrorHostMemoryAlreadyRegistered = \
        <underlying_type_error> cudaError.cudaErrorHostMemoryAlreadyRegistered
    cudaErrorHostMemoryNotRegistered = \
        <underlying_type_error> cudaError.cudaErrorHostMemoryNotRegistered
    cudaErrorHardwareStackError = \
        <underlying_type_error> cudaError.cudaErrorHardwareStackError
    cudaErrorIllegalInstruction = \
        <underlying_type_error> cudaError.cudaErrorIllegalInstruction
    cudaErrorMisalignedAddress = \
        <underlying_type_error> cudaError.cudaErrorMisalignedAddress
    cudaErrorInvalidAddressSpace = \
        <underlying_type_error> cudaError.cudaErrorInvalidAddressSpace
    cudaErrorInvalidPc = \
        <underlying_type_error> cudaError.cudaErrorInvalidPc
    cudaErrorLaunchFailure = \
        <underlying_type_error> cudaError.cudaErrorLaunchFailure
    cudaErrorCooperativeLaunchTooLarge = \
        <underlying_type_error> cudaError.cudaErrorCooperativeLaunchTooLarge
    cudaErrorNotPermitted = \
        <underlying_type_error> cudaError.cudaErrorNotPermitted
    cudaErrorNotSupported = \
        <underlying_type_error> cudaError.cudaErrorNotSupported
    cudaErrorSystemNotReady = \
        <underlying_type_error> cudaError.cudaErrorSystemNotReady
    cudaErrorSystemDriverMismatch = \
        <underlying_type_error> cudaError.cudaErrorSystemDriverMismatch
    cudaErrorCompatNotSupportedOnDevice = \
        <underlying_type_error> cudaError.cudaErrorCompatNotSupportedOnDevice
    cudaErrorStreamCaptureUnsupported = \
        <underlying_type_error> cudaError.cudaErrorStreamCaptureUnsupported
    cudaErrorStreamCaptureInvalidated = \
        <underlying_type_error> cudaError.cudaErrorStreamCaptureInvalidated
    cudaErrorStreamCaptureMerge = \
        <underlying_type_error> cudaError.cudaErrorStreamCaptureMerge
    cudaErrorStreamCaptureUnmatched = \
        <underlying_type_error> cudaError.cudaErrorStreamCaptureUnmatched
    cudaErrorStreamCaptureUnjoined = \
        <underlying_type_error> cudaError.cudaErrorStreamCaptureUnjoined
    cudaErrorStreamCaptureIsolation = \
        <underlying_type_error> cudaError.cudaErrorStreamCaptureIsolation
    cudaErrorStreamCaptureImplicit = \
        <underlying_type_error> cudaError.cudaErrorStreamCaptureImplicit
    cudaErrorCapturedEvent = \
        <underlying_type_error> cudaError.cudaErrorCapturedEvent
    cudaErrorStreamCaptureWrongThread = \
        <underlying_type_error> cudaError.cudaErrorStreamCaptureWrongThread
    cudaErrorTimeout = \
        <underlying_type_error> cudaError.cudaErrorTimeout
    cudaErrorGraphExecUpdateFailure = \
        <underlying_type_error> cudaError.cudaErrorGraphExecUpdateFailure
    cudaErrorUnknown = \
        <underlying_type_error> cudaError.cudaErrorUnknown
    cudaErrorApiFailureBase = \
        <underlying_type_error> cudaError.cudaErrorApiFailureBase


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
