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

    int cuDeviceGetName(char* name, int length, int device)

cdef extern from "cuda_runtime_api.h" nogil:

    int cudaDriverGetVersion(int* driverVersion)
    int cudaRuntimeGetVersion(int* runtimeVersion)
    int cudaGetDeviceCount(int* count)
    int cudaDeviceGetAttribute(int* value, cudaDeviceAttr attr, int device)
    int cudaGetDeviceProperties(cudaDeviceProp* prop, int device)

    const char* cudaGetErrorName(cudaError_t error)
    const char* cudaGetErrorString(cudaError_t error)

ctypedef int underlying_type_attribute
