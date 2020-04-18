# Copyright (c) 2020, NVIDIA CORPORATION.

cdef extern from "cuda.h" nogil:
    cdef enum cudaDeviceAttr:
        cudaDevAttrMaxThreadsPerBlock = 1
        cudaDevAttrMaxBlockDimX = 2
        cudaDevAttrMaxBlockDimY = 3
        cudaDevAttrMaxBlockDimZ = 4
        cudaDevAttrMaxGridDimX = 5
        cudaDevAttrMaxGridDimY = 6
        cudaDevAttrMaxGridDimZ = 7
        cudaDevAttrMaxSharedMemoryPerBlock = 8
        cudaDevAttrTotalConstantMemory = 9
        cudaDevAttrWarpSize = 10
        cudaDevAttrMaxPitch = 11
        cudaDevAttrMaxRegistersPerBlock = 12
        cudaDevAttrClockRate = 13
        cudaDevAttrTextureAlignment = 14
        cudaDevAttrGpuOverlap = 15
        cudaDevAttrMultiProcessorCount = 16
        cudaDevAttrKernelExecTimeout = 17
        cudaDevAttrIntegrated = 18
        cudaDevAttrCanMapHostMemory = 19
        cudaDevAttrComputeMode = 20
        cudaDevAttrMaxTexture1DWidth = 21
        cudaDevAttrMaxTexture2DWidth = 22
        cudaDevAttrMaxTexture2DHeight = 23
        cudaDevAttrMaxTexture3DWidth = 24
        cudaDevAttrMaxTexture3DHeight = 25
        cudaDevAttrMaxTexture3DDepth = 26
        cudaDevAttrMaxTexture2DLayeredWidth = 27
        cudaDevAttrMaxTexture2DLayeredHeight = 28
        cudaDevAttrMaxTexture2DLayeredLayers = 29
        cudaDevAttrSurfaceAlignment = 30
        cudaDevAttrConcurrentKernels = 31
        cudaDevAttrEccEnabled = 32
        cudaDevAttrPciBusId = 33
        cudaDevAttrPciDeviceId = 34
        cudaDevAttrTccDriver = 35
        cudaDevAttrMemoryClockRate = 36
        cudaDevAttrGlobalMemoryBusWidth = 37
        cudaDevAttrL2CacheSize = 38
        cudaDevAttrMaxThreadsPerMultiProcessor = 39
        cudaDevAttrAsyncEngineCount = 40
        cudaDevAttrUnifiedAddressing = 41
        cudaDevAttrMaxTexture1DLayeredWidth = 42
        cudaDevAttrMaxTexture1DLayeredLayers = 43
        cudaDevAttrMaxTexture2DGatherWidth = 45
        cudaDevAttrMaxTexture2DGatherHeight = 46
        cudaDevAttrMaxTexture3DWidthAlt = 47
        cudaDevAttrMaxTexture3DHeightAlt = 48
        cudaDevAttrMaxTexture3DDepthAlt = 49
        cudaDevAttrPciDomainId = 50
        cudaDevAttrTexturePitchAlignment = 51
        cudaDevAttrMaxTextureCubemapWidth = 52
        cudaDevAttrMaxTextureCubemapLayeredWidth = 53
        cudaDevAttrMaxTextureCubemapLayeredLayers = 54
        cudaDevAttrMaxSurface1DWidth = 55
        cudaDevAttrMaxSurface2DWidth = 56
        cudaDevAttrMaxSurface2DHeight = 57
        cudaDevAttrMaxSurface3DWidth = 58
        cudaDevAttrMaxSurface3DHeight = 59
        cudaDevAttrMaxSurface3DDepth = 60
        cudaDevAttrMaxSurface1DLayeredWidth = 61
        cudaDevAttrMaxSurface1DLayeredLayers = 62
        cudaDevAttrMaxSurface2DLayeredWidth = 63
        cudaDevAttrMaxSurface2DLayeredHeight = 64
        cudaDevAttrMaxSurface2DLayeredLayers = 65
        cudaDevAttrMaxSurfaceCubemapWidth = 66
        cudaDevAttrMaxSurfaceCubemapLayeredWidth = 67
        cudaDevAttrMaxSurfaceCubemapLayeredLayers = 68
        cudaDevAttrMaxTexture1DLinearWidth = 69
        cudaDevAttrMaxTexture2DLinearWidth = 70
        cudaDevAttrMaxTexture2DLinearHeight = 71
        cudaDevAttrMaxTexture2DLinearPitch = 72
        cudaDevAttrMaxTexture2DMipmappedWidth = 73
        cudaDevAttrMaxTexture2DMipmappedHeight = 74
        cudaDevAttrComputeCapabilityMajor = 75
        cudaDevAttrComputeCapabilityMinor = 76
        cudaDevAttrMaxTexture1DMipmappedWidth = 77
        cudaDevAttrStreamPrioritiesSupported = 78
        cudaDevAttrGlobalL1CacheSupported = 79
        cudaDevAttrLocalL1CacheSupported = 80
        cudaDevAttrMaxSharedMemoryPerMultiprocessor = 81
        cudaDevAttrMaxRegistersPerMultiprocessor = 82
        cudaDevAttrManagedMemory = 83
        cudaDevAttrIsMultiGpuBoard = 84
        cudaDevAttrMultiGpuBoardGroupID = 85
        cudaDevAttrHostNativeAtomicSupported = 86
        cudaDevAttrSingleToDoublePrecisionPerfRatio = 87
        cudaDevAttrPageableMemoryAccess = 88
        cudaDevAttrConcurrentManagedAccess = 89
        cudaDevAttrComputePreemptionSupported = 90
        cudaDevAttrCanUseHostPointerForRegisteredMem = 91
        cudaDevAttrReserved92 = 92
        cudaDevAttrReserved93 = 93
        cudaDevAttrReserved94 = 94
        cudaDevAttrCooperativeLaunch = 95
        cudaDevAttrCooperativeMultiDeviceLaunch = 96
        cudaDevAttrMaxSharedMemoryPerBlockOptin = 97
        cudaDevAttrCanFlushRemoteWrites = 98
        cudaDevAttrHostRegisterSupported = 99
        cudaDevAttrPageableMemoryAccessUsesHostPageTables = 100
        cudaDevAttrDirectManagedMemAccessFromHost = 101

    ctypedef enum cudaError:
        cudaSuccess = 0
        cudaErrorInvalidValue = 1
        cudaErrorMemoryAllocation = 2
        cudaErrorInitializationError = 3
        cudaErrorCudartUnloading = 4
        cudaErrorProfilerDisabled = 5
        cudaErrorProfilerNotInitialized = 6
        cudaErrorProfilerAlreadyStarted = 7
        cudaErrorProfilerAlreadyStopped = 8
        cudaErrorInvalidConfiguration = 9
        cudaErrorInvalidPitchValue = 12
        cudaErrorInvalidSymbol = 13
        cudaErrorInvalidHostPointer = 16
        cudaErrorInvalidDevicePointer = 17
        cudaErrorInvalidTexture = 18
        cudaErrorInvalidTextureBinding = 19
        cudaErrorInvalidChannelDescriptor = 20
        cudaErrorInvalidMemcpyDirection = 21
        cudaErrorAddressOfConstant = 22
        cudaErrorTextureFetchFailed = 23
        cudaErrorTextureNotBound = 24
        cudaErrorSynchronizationError = 25
        cudaErrorInvalidFilterSetting = 26
        cudaErrorInvalidNormSetting = 27
        cudaErrorMixedDeviceExecution = 28
        cudaErrorNotYetImplemented = 31
        cudaErrorMemoryValueTooLarge = 32
        cudaErrorInsufficientDriver = 35
        cudaErrorInvalidSurface = 37
        cudaErrorDuplicateVariableName = 43
        cudaErrorDuplicateTextureName = 44
        cudaErrorDuplicateSurfaceName = 45
        cudaErrorDevicesUnavailable = 46
        cudaErrorIncompatibleDriverContext = 49
        cudaErrorMissingConfiguration = 52
        cudaErrorPriorLaunchFailure = 53
        cudaErrorLaunchMaxDepthExceeded = 65
        cudaErrorLaunchFileScopedTex = 66
        cudaErrorLaunchFileScopedSurf = 67
        cudaErrorSyncDepthExceeded = 68
        cudaErrorLaunchPendingCountExceeded = 69
        cudaErrorInvalidDeviceFunction = 98
        cudaErrorNoDevice = 100
        cudaErrorInvalidDevice = 101
        cudaErrorStartupFailure = 127
        cudaErrorInvalidKernelImage = 200
        cudaErrorDeviceUninitialized = 201
        cudaErrorMapBufferObjectFailed = 205
        cudaErrorUnmapBufferObjectFailed = 206
        cudaErrorArrayIsMapped = 207
        cudaErrorAlreadyMapped = 208
        cudaErrorNoKernelImageForDevice = 209
        cudaErrorAlreadyAcquired = 210
        cudaErrorNotMapped = 211
        cudaErrorNotMappedAsArray = 212
        cudaErrorNotMappedAsPointer = 213
        cudaErrorECCUncorrectable = 214
        cudaErrorUnsupportedLimit = 215
        cudaErrorDeviceAlreadyInUse = 216
        cudaErrorPeerAccessUnsupported = 217
        cudaErrorInvalidPtx = 218
        cudaErrorInvalidGraphicsContext = 219
        cudaErrorNvlinkUncorrectable = 220
        cudaErrorJitCompilerNotFound = 221
        cudaErrorInvalidSource = 300
        cudaErrorFileNotFound = 301
        cudaErrorSharedObjectSymbolNotFound = 302
        cudaErrorSharedObjectInitFailed = 303
        cudaErrorOperatingSystem = 304
        cudaErrorInvalidResourceHandle = 400
        cudaErrorIllegalState = 401
        cudaErrorSymbolNotFound = 500
        cudaErrorNotReady = 600
        cudaErrorIllegalAddress = 700
        cudaErrorLaunchOutOfResources = 701
        cudaErrorLaunchTimeout = 702
        cudaErrorLaunchIncompatibleTexturing = 703
        cudaErrorPeerAccessAlreadyEnabled = 704
        cudaErrorPeerAccessNotEnabled = 705
        cudaErrorSetOnActiveProcess = 708
        cudaErrorContextIsDestroyed = 709
        cudaErrorAssert = 710
        cudaErrorTooManyPeers = 711
        cudaErrorHostMemoryAlreadyRegistered = 712
        cudaErrorHostMemoryNotRegistered = 713
        cudaErrorHardwareStackError = 714
        cudaErrorIllegalInstruction = 715
        cudaErrorMisalignedAddress = 716
        cudaErrorInvalidAddressSpace = 717
        cudaErrorInvalidPc = 718
        cudaErrorLaunchFailure = 719
        cudaErrorCooperativeLaunchTooLarge = 720
        cudaErrorNotPermitted = 800
        cudaErrorNotSupported = 801
        cudaErrorSystemNotReady = 802
        cudaErrorSystemDriverMismatch = 803
        cudaErrorCompatNotSupportedOnDevice = 804
        cudaErrorStreamCaptureUnsupported = 900
        cudaErrorStreamCaptureInvalidated = 901
        cudaErrorStreamCaptureMerge = 902
        cudaErrorStreamCaptureUnmatched = 903
        cudaErrorStreamCaptureUnjoined = 904
        cudaErrorStreamCaptureIsolation = 905
        cudaErrorStreamCaptureImplicit = 906
        cudaErrorCapturedEvent = 907
        cudaErrorStreamCaptureWrongThread = 908
        cudaErrorTimeout = 909
        cudaErrorGraphExecUpdateFailure = 910
        cudaErrorUnknown = 999
        cudaErrorApiFailureBase = 10000

    ctypedef cudaError cudaError_t

    ctypedef enum CUresult:
        CUDA_SUCCESS = 0
        CUDA_ERROR_INVALID_VALUE = 1
        CUDA_ERROR_OUT_OF_MEMORY = 2
        CUDA_ERROR_NOT_INITIALIZED = 3
        CUDA_ERROR_DEINITIALIZED = 4
        CUDA_ERROR_PROFILER_DISABLED = 5
        CUDA_ERROR_PROFILER_NOT_INITIALIZED = 6
        CUDA_ERROR_PROFILER_ALREADY_STARTED = 7
        CUDA_ERROR_PROFILER_ALREADY_STOPPED = 8
        CUDA_ERROR_NO_DEVICE = 100
        CUDA_ERROR_INVALID_DEVICE = 101
        CUDA_ERROR_INVALID_IMAGE = 200
        CUDA_ERROR_INVALID_CONTEXT = 201
        CUDA_ERROR_CONTEXT_ALREADY_CURRENT = 202
        CUDA_ERROR_MAP_FAILED = 205
        CUDA_ERROR_UNMAP_FAILED = 206
        CUDA_ERROR_ARRAY_IS_MAPPED = 207
        CUDA_ERROR_ALREADY_MAPPED = 208
        CUDA_ERROR_NO_BINARY_FOR_GPU = 209
        CUDA_ERROR_ALREADY_ACQUIRED = 210
        CUDA_ERROR_NOT_MAPPED = 211
        CUDA_ERROR_NOT_MAPPED_AS_ARRAY = 212
        CUDA_ERROR_NOT_MAPPED_AS_POINTER = 213
        CUDA_ERROR_ECC_UNCORRECTABLE = 214
        CUDA_ERROR_UNSUPPORTED_LIMIT = 215
        CUDA_ERROR_CONTEXT_ALREADY_IN_USE = 216
        CUDA_ERROR_PEER_ACCESS_UNSUPPORTED = 217
        CUDA_ERROR_INVALID_PTX = 218
        CUDA_ERROR_INVALID_GRAPHICS_CONTEXT = 219
        CUDA_ERROR_NVLINK_UNCORRECTABLE = 220
        CUDA_ERROR_JIT_COMPILER_NOT_FOUND = 221
        CUDA_ERROR_INVALID_SOURCE = 300
        CUDA_ERROR_FILE_NOT_FOUND = 301
        CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND = 302
        CUDA_ERROR_SHARED_OBJECT_INIT_FAILED = 303
        CUDA_ERROR_OPERATING_SYSTEM = 304
        CUDA_ERROR_INVALID_HANDLE = 400
        CUDA_ERROR_ILLEGAL_STATE = 401
        CUDA_ERROR_NOT_FOUND = 500
        CUDA_ERROR_NOT_READY = 600
        CUDA_ERROR_ILLEGAL_ADDRESS = 700
        CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES = 701
        CUDA_ERROR_LAUNCH_TIMEOUT = 702
        CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING = 703
        CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED = 704
        CUDA_ERROR_PEER_ACCESS_NOT_ENABLED = 705
        CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE = 708
        CUDA_ERROR_CONTEXT_IS_DESTROYED = 709
        CUDA_ERROR_ASSERT = 710
        CUDA_ERROR_TOO_MANY_PEERS = 711
        CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED = 712
        CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED = 713
        CUDA_ERROR_HARDWARE_STACK_ERROR = 714
        CUDA_ERROR_ILLEGAL_INSTRUCTION = 715
        CUDA_ERROR_MISALIGNED_ADDRESS = 716
        CUDA_ERROR_INVALID_ADDRESS_SPACE = 717
        CUDA_ERROR_INVALID_PC = 718
        CUDA_ERROR_LAUNCH_FAILED = 719
        CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE = 720
        CUDA_ERROR_NOT_PERMITTED = 800
        CUDA_ERROR_NOT_SUPPORTED = 801
        CUDA_ERROR_SYSTEM_NOT_READY = 802
        CUDA_ERROR_SYSTEM_DRIVER_MISMATCH = 803
        CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE = 804
        CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED = 900
        CUDA_ERROR_STREAM_CAPTURE_INVALIDATED = 901
        CUDA_ERROR_STREAM_CAPTURE_MERGE = 902
        CUDA_ERROR_STREAM_CAPTURE_UNMATCHED = 903
        CUDA_ERROR_STREAM_CAPTURE_UNJOINED = 904
        CUDA_ERROR_STREAM_CAPTURE_ISOLATION = 905
        CUDA_ERROR_STREAM_CAPTURE_IMPLICIT = 906
        CUDA_ERROR_CAPTURED_EVENT = 907
        CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD = 908
        CUDA_ERROR_TIMEOUT = 909
        CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE = 910
        CUDA_ERROR_UNKNOWN = 999

    ctypedef struct CUuuid_st:
        char  bytes[16]

    ctypedef CUuuid_st cudaUUID_t

    ctypedef struct cudaDeviceProp:
        int  ECCEnabled
        int  asyncEngineCount
        int  canMapHostMemory
        int  canUseHostPointerForRegisteredMem
        int  clockRate
        int  computeMode
        int  computePreemptionSupported
        int  concurrentKernels
        int  concurrentManagedAccess
        int  cooperativeLaunch
        int  cooperativeMultiDeviceLaunch
        int  deviceOverlap
        int  directManagedMemAccessFromHost
        int  globalL1CacheSupported
        int  hostNativeAtomicSupported
        int  integrated
        int  isMultiGpuBoard
        int  kernelExecTimeoutEnabled
        int  l2CacheSize
        int  localL1CacheSupported
        char  luid[8]
        unsigned int  luidDeviceNodeMask
        int  major
        int  managedMemory
        int  maxGridSize[3]
        int  maxSurface1D
        int  maxSurface1DLayered[2]
        int  maxSurface2D[2]
        int  maxSurface2DLayered[3]
        int  maxSurface3D[3]
        int  maxSurfaceCubemap
        int  maxSurfaceCubemapLayered[2]
        int  maxTexture1D
        int  maxTexture1DLayered[2]
        int  maxTexture1DLinear
        int  maxTexture1DMipmap
        int  maxTexture2D[2]
        int  maxTexture2DGather[2]
        int  maxTexture2DLayered[3]
        int  maxTexture2DLinear[3]
        int  maxTexture2DMipmap[2]
        int  maxTexture3D[3]
        int  maxTexture3DAlt[3]
        int  maxTextureCubemap
        int  maxTextureCubemapLayered[2]
        int  maxThreadsDim[3]
        int  maxThreadsPerBlock
        int  maxThreadsPerMultiProcessor
        size_t  memPitch
        int  memoryBusWidth
        int  memoryClockRate
        int  minor
        int  multiGpuBoardGroupID
        int  multiProcessorCount
        char  name[256]
        int  pageableMemoryAccess
        int  pageableMemoryAccessUsesHostPageTables
        int  pciBusID
        int  pciDeviceID
        int  pciDomainID
        int  regsPerBlock
        int  regsPerMultiprocessor
        size_t  sharedMemPerBlock
        size_t  sharedMemPerBlockOptin
        size_t  sharedMemPerMultiprocessor
        int  singleToDoublePrecisionPerfRatio
        int  streamPrioritiesSupported
        size_t  surfaceAlignment
        int  tccDriver
        size_t  textureAlignment
        size_t  texturePitchAlignment
        size_t  totalConstMem
        size_t  totalGlobalMem
        int  unifiedAddressing
        cudaUUID_t  uuid
        int  warpSize

    CUresult cuDeviceGetName(char* name, int length, int device)

cdef extern from "cuda_runtime_api.h" nogil:

    cudaError_t cudaDriverGetVersion(int* driverVersion)
    cudaError_t cudaRuntimeGetVersion(int* runtimeVersion)
    cudaError_t cudaGetDeviceCount(int* count)
    cudaError_t cudaDeviceGetAttribute(int* value,
                                       cudaDeviceAttr attr,
                                       int device)
    cudaError_t cudaGetDeviceProperties(cudaDeviceProp* prop, int device)

    const char* cudaGetErrorName(cudaError_t error)
    const char* cudaGetErrorString(cudaError_t error)

ctypedef int underlying_type_attribute
