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

cdef extern from "cuda_runtime_api.h" nogil:

    int cudaDriverGetVersion(int* driverVersion) except +
    int cudaRuntimeGetVersion(int* runtimeVersion) except +
    int cudaGetDeviceCount(int* count) except +
    int cudaDeviceGetAttribute(int* value, cudaDeviceAttr attr, int device) \
        except +

ctypedef int underlying_type_attribute
