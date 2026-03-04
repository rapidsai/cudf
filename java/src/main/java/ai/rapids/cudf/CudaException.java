/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2023, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
package ai.rapids.cudf;

import java.util.HashMap;
import java.util.Map;

/**
 * Exception from the cuda language/library. Be aware that because of how cuda does asynchronous
 * processing exceptions from cuda can be thrown by method calls that did not cause the exception
 * to take place. These will take place on the same thread that caused the error.
 * <p>
 * Please See
 * <a href="https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__ERROR.html">the cuda docs</a>
 * for more details on how this works.
 * <p>
 * In general you can recover from cuda errors even in async calls if you make sure that you
 * don't switch between threads for different parts of processing that can be retried as a chunk.
 */
public class CudaException extends RuntimeException {
  CudaException(String message, int errorCode) {
    this(message, "No native stacktrace is available.", errorCode);
  }

  CudaException(String message, String nativeStacktrace, int errorCode) {
    super(message);
    this.nativeStacktrace = nativeStacktrace;
    cudaError = CudaError.parseErrorCode(errorCode);
  }

  CudaException(String message, String nativeStacktrace, int errorCode, Throwable cause) {
    super(message, cause);
    this.nativeStacktrace = nativeStacktrace;
    cudaError = CudaError.parseErrorCode(errorCode);
  }

  public String getNativeStacktrace() {
    return nativeStacktrace;
  }

  public CudaError getCudaError() {
    return cudaError;
  }

  private final String nativeStacktrace;

  private final CudaError cudaError;

  /**
   * The Java mirror of cudaError, which facilities the tracking of CUDA errors in JVM.
   */
  public enum CudaError {
    UnknownNativeError(-1), // native CUDA error type which Java doesn't have a representation
    cudaErrorInvalidValue(1),
    cudaErrorMemoryAllocation(2),
    cudaErrorInitializationError(3),
    cudaErrorCudartUnloading(4),
    cudaErrorProfilerDisabled(5),
    cudaErrorProfilerNotInitialized(6),
    cudaErrorProfilerAlreadyStarted(7),
    cudaErrorProfilerAlreadyStopped(8),
    cudaErrorInvalidConfiguration(9),
    cudaErrorInvalidPitchValue(12),
    cudaErrorInvalidSymbol(13),
    cudaErrorInvalidHostPointer(16),
    cudaErrorInvalidDevicePointer(17),
    cudaErrorInvalidTexture(18),
    cudaErrorInvalidTextureBinding(19),
    cudaErrorInvalidChannelDescriptor(20),
    cudaErrorInvalidMemcpyDirection(21),
    cudaErrorAddressOfConstant(22),
    cudaErrorTextureFetchFailed(23),
    cudaErrorTextureNotBound(24),
    cudaErrorSynchronizationError(25),
    cudaErrorInvalidFilterSetting(26),
    cudaErrorInvalidNormSetting(27),
    cudaErrorMixedDeviceExecution(28),
    cudaErrorNotYetImplemented(31),
    cudaErrorMemoryValueTooLarge(32),
    cudaErrorStubLibrary(34),
    cudaErrorInsufficientDriver(35),
    cudaErrorCallRequiresNewerDriver(36),
    cudaErrorInvalidSurface(37),
    cudaErrorDuplicateVariableName(43),
    cudaErrorDuplicateTextureName(44),
    cudaErrorDuplicateSurfaceName(45),
    cudaErrorDevicesUnavailable(46),
    cudaErrorIncompatibleDriverContext(49),
    cudaErrorMissingConfiguration(52),
    cudaErrorPriorLaunchFailure(53),
    cudaErrorLaunchMaxDepthExceeded(65),
    cudaErrorLaunchFileScopedTex(66),
    cudaErrorLaunchFileScopedSurf(67),
    cudaErrorSyncDepthExceeded(68),
    cudaErrorLaunchPendingCountExceeded(69),
    cudaErrorInvalidDeviceFunction(98),
    cudaErrorNoDevice(100),
    cudaErrorInvalidDevice(101),
    cudaErrorDeviceNotLicensed(102),
    cudaErrorSoftwareValidityNotEstablished(103),
    cudaErrorStartupFailure(127),
    cudaErrorInvalidKernelImage(200),
    cudaErrorDeviceUninitialized(201),
    cudaErrorMapBufferObjectFailed(205),
    cudaErrorUnmapBufferObjectFailed(206),
    cudaErrorArrayIsMapped(207),
    cudaErrorAlreadyMapped(208),
    cudaErrorNoKernelImageForDevice(209),
    cudaErrorAlreadyAcquired(210),
    cudaErrorNotMapped(211),
    cudaErrorNotMappedAsArray(212),
    cudaErrorNotMappedAsPointer(213),
    cudaErrorECCUncorrectable(214),
    cudaErrorUnsupportedLimit(215),
    cudaErrorDeviceAlreadyInUse(216),
    cudaErrorPeerAccessUnsupported(217),
    cudaErrorInvalidPtx(218),
    cudaErrorInvalidGraphicsContext(219),
    cudaErrorNvlinkUncorrectable(220),
    cudaErrorJitCompilerNotFound(221),
    cudaErrorUnsupportedPtxVersion(222),
    cudaErrorJitCompilationDisabled(223),
    cudaErrorUnsupportedExecAffinity(224),
    cudaErrorInvalidSource(300),
    cudaErrorFileNotFound(301),
    cudaErrorSharedObjectSymbolNotFound(302),
    cudaErrorSharedObjectInitFailed(303),
    cudaErrorOperatingSystem(304),
    cudaErrorInvalidResourceHandle(400),
    cudaErrorIllegalState(401),
    cudaErrorSymbolNotFound(500),
    cudaErrorNotReady(600),
    cudaErrorIllegalAddress(700),
    cudaErrorLaunchOutOfResources(701),
    cudaErrorLaunchTimeout(702),
    cudaErrorLaunchIncompatibleTexturing(703),
    cudaErrorPeerAccessAlreadyEnabled(704),
    cudaErrorPeerAccessNotEnabled(705),
    cudaErrorSetOnActiveProcess(708),
    cudaErrorContextIsDestroyed(709),
    cudaErrorAssert(710),
    cudaErrorTooManyPeers(711),
    cudaErrorHostMemoryAlreadyRegistered(712),
    cudaErrorHostMemoryNotRegistered(713),
    cudaErrorHardwareStackError(714),
    cudaErrorIllegalInstruction(715),
    cudaErrorMisalignedAddress(716),
    cudaErrorInvalidAddressSpace(717),
    cudaErrorInvalidPc(718),
    cudaErrorLaunchFailure(719),
    cudaErrorCooperativeLaunchTooLarge(720),
    cudaErrorNotPermitted(800),
    cudaErrorNotSupported(801),
    cudaErrorSystemNotReady(802),
    cudaErrorSystemDriverMismatch(803),
    cudaErrorCompatNotSupportedOnDevice(804),
    cudaErrorMpsConnectionFailed(805),
    cudaErrorMpsRpcFailure(806),
    cudaErrorMpsServerNotReady(807),
    cudaErrorMpsMaxClientsReached(808),
    cudaErrorMpsMaxConnectionsReached(809),
    cudaErrorStreamCaptureUnsupported(900),
    cudaErrorStreamCaptureInvalidated(901),
    cudaErrorStreamCaptureMerge(902),
    cudaErrorStreamCaptureUnmatched(903),
    cudaErrorStreamCaptureUnjoined(904),
    cudaErrorStreamCaptureIsolation(905),
    cudaErrorStreamCaptureImplicit(906),
    cudaErrorCapturedEvent(907),
    cudaErrorStreamCaptureWrongThread(908),
    cudaErrorTimeout(909),
    cudaErrorGraphExecUpdateFailure(910),
    cudaErrorExternalDevice(911),
    cudaErrorUnknown(999),
    cudaErrorApiFailureBase(10000);

    final int code;

    private static Map<Integer, CudaError> codeToError = new HashMap<Integer, CudaError>(){{
      put(cudaErrorInvalidValue.code, cudaErrorInvalidValue);
      put(cudaErrorMemoryAllocation.code, cudaErrorMemoryAllocation);
      put(cudaErrorInitializationError.code, cudaErrorInitializationError);
      put(cudaErrorCudartUnloading.code, cudaErrorCudartUnloading);
      put(cudaErrorProfilerDisabled.code, cudaErrorProfilerDisabled);
      put(cudaErrorProfilerNotInitialized.code, cudaErrorProfilerNotInitialized);
      put(cudaErrorProfilerAlreadyStarted.code, cudaErrorProfilerAlreadyStarted);
      put(cudaErrorProfilerAlreadyStopped.code, cudaErrorProfilerAlreadyStopped);
      put(cudaErrorInvalidConfiguration.code, cudaErrorInvalidConfiguration);
      put(cudaErrorInvalidPitchValue.code, cudaErrorInvalidPitchValue);
      put(cudaErrorInvalidSymbol.code, cudaErrorInvalidSymbol);
      put(cudaErrorInvalidHostPointer.code, cudaErrorInvalidHostPointer);
      put(cudaErrorInvalidDevicePointer.code, cudaErrorInvalidDevicePointer);
      put(cudaErrorInvalidTexture.code, cudaErrorInvalidTexture);
      put(cudaErrorInvalidTextureBinding.code, cudaErrorInvalidTextureBinding);
      put(cudaErrorInvalidChannelDescriptor.code, cudaErrorInvalidChannelDescriptor);
      put(cudaErrorInvalidMemcpyDirection.code, cudaErrorInvalidMemcpyDirection);
      put(cudaErrorAddressOfConstant.code, cudaErrorAddressOfConstant);
      put(cudaErrorTextureFetchFailed.code, cudaErrorTextureFetchFailed);
      put(cudaErrorTextureNotBound.code, cudaErrorTextureNotBound);
      put(cudaErrorSynchronizationError.code, cudaErrorSynchronizationError);
      put(cudaErrorInvalidFilterSetting.code, cudaErrorInvalidFilterSetting);
      put(cudaErrorInvalidNormSetting.code, cudaErrorInvalidNormSetting);
      put(cudaErrorMixedDeviceExecution.code, cudaErrorMixedDeviceExecution);
      put(cudaErrorNotYetImplemented.code, cudaErrorNotYetImplemented);
      put(cudaErrorMemoryValueTooLarge.code, cudaErrorMemoryValueTooLarge);
      put(cudaErrorStubLibrary.code, cudaErrorStubLibrary);
      put(cudaErrorInsufficientDriver.code, cudaErrorInsufficientDriver);
      put(cudaErrorCallRequiresNewerDriver.code, cudaErrorCallRequiresNewerDriver);
      put(cudaErrorInvalidSurface.code, cudaErrorInvalidSurface);
      put(cudaErrorDuplicateVariableName.code, cudaErrorDuplicateVariableName);
      put(cudaErrorDuplicateTextureName.code, cudaErrorDuplicateTextureName);
      put(cudaErrorDuplicateSurfaceName.code, cudaErrorDuplicateSurfaceName);
      put(cudaErrorDevicesUnavailable.code, cudaErrorDevicesUnavailable);
      put(cudaErrorIncompatibleDriverContext.code, cudaErrorIncompatibleDriverContext);
      put(cudaErrorMissingConfiguration.code, cudaErrorMissingConfiguration);
      put(cudaErrorPriorLaunchFailure.code, cudaErrorPriorLaunchFailure);
      put(cudaErrorLaunchMaxDepthExceeded.code, cudaErrorLaunchMaxDepthExceeded);
      put(cudaErrorLaunchFileScopedTex.code, cudaErrorLaunchFileScopedTex);
      put(cudaErrorLaunchFileScopedSurf.code, cudaErrorLaunchFileScopedSurf);
      put(cudaErrorSyncDepthExceeded.code, cudaErrorSyncDepthExceeded);
      put(cudaErrorLaunchPendingCountExceeded.code, cudaErrorLaunchPendingCountExceeded);
      put(cudaErrorInvalidDeviceFunction.code, cudaErrorInvalidDeviceFunction);
      put(cudaErrorNoDevice.code, cudaErrorNoDevice);
      put(cudaErrorInvalidDevice.code, cudaErrorInvalidDevice);
      put(cudaErrorDeviceNotLicensed.code, cudaErrorDeviceNotLicensed);
      put(cudaErrorSoftwareValidityNotEstablished.code, cudaErrorSoftwareValidityNotEstablished);
      put(cudaErrorStartupFailure.code, cudaErrorStartupFailure);
      put(cudaErrorInvalidKernelImage.code, cudaErrorInvalidKernelImage);
      put(cudaErrorDeviceUninitialized.code, cudaErrorDeviceUninitialized);
      put(cudaErrorMapBufferObjectFailed.code, cudaErrorMapBufferObjectFailed);
      put(cudaErrorUnmapBufferObjectFailed.code, cudaErrorUnmapBufferObjectFailed);
      put(cudaErrorArrayIsMapped.code, cudaErrorArrayIsMapped);
      put(cudaErrorAlreadyMapped.code, cudaErrorAlreadyMapped);
      put(cudaErrorNoKernelImageForDevice.code, cudaErrorNoKernelImageForDevice);
      put(cudaErrorAlreadyAcquired.code, cudaErrorAlreadyAcquired);
      put(cudaErrorNotMapped.code, cudaErrorNotMapped);
      put(cudaErrorNotMappedAsArray.code, cudaErrorNotMappedAsArray);
      put(cudaErrorNotMappedAsPointer.code, cudaErrorNotMappedAsPointer);
      put(cudaErrorECCUncorrectable.code, cudaErrorECCUncorrectable);
      put(cudaErrorUnsupportedLimit.code, cudaErrorUnsupportedLimit);
      put(cudaErrorDeviceAlreadyInUse.code, cudaErrorDeviceAlreadyInUse);
      put(cudaErrorPeerAccessUnsupported.code, cudaErrorPeerAccessUnsupported);
      put(cudaErrorInvalidPtx.code, cudaErrorInvalidPtx);
      put(cudaErrorInvalidGraphicsContext.code, cudaErrorInvalidGraphicsContext);
      put(cudaErrorNvlinkUncorrectable.code, cudaErrorNvlinkUncorrectable);
      put(cudaErrorJitCompilerNotFound.code, cudaErrorJitCompilerNotFound);
      put(cudaErrorUnsupportedPtxVersion.code, cudaErrorUnsupportedPtxVersion);
      put(cudaErrorJitCompilationDisabled.code, cudaErrorJitCompilationDisabled);
      put(cudaErrorUnsupportedExecAffinity.code, cudaErrorUnsupportedExecAffinity);
      put(cudaErrorInvalidSource.code, cudaErrorInvalidSource);
      put(cudaErrorFileNotFound.code, cudaErrorFileNotFound);
      put(cudaErrorSharedObjectSymbolNotFound.code, cudaErrorSharedObjectSymbolNotFound);
      put(cudaErrorSharedObjectInitFailed.code, cudaErrorSharedObjectInitFailed);
      put(cudaErrorOperatingSystem.code, cudaErrorOperatingSystem);
      put(cudaErrorInvalidResourceHandle.code, cudaErrorInvalidResourceHandle);
      put(cudaErrorIllegalState.code, cudaErrorIllegalState);
      put(cudaErrorSymbolNotFound.code, cudaErrorSymbolNotFound);
      put(cudaErrorNotReady.code, cudaErrorNotReady);
      put(cudaErrorIllegalAddress.code, cudaErrorIllegalAddress);
      put(cudaErrorLaunchOutOfResources.code, cudaErrorLaunchOutOfResources);
      put(cudaErrorLaunchTimeout.code, cudaErrorLaunchTimeout);
      put(cudaErrorLaunchIncompatibleTexturing.code, cudaErrorLaunchIncompatibleTexturing);
      put(cudaErrorPeerAccessAlreadyEnabled.code, cudaErrorPeerAccessAlreadyEnabled);
      put(cudaErrorPeerAccessNotEnabled.code, cudaErrorPeerAccessNotEnabled);
      put(cudaErrorSetOnActiveProcess.code, cudaErrorSetOnActiveProcess);
      put(cudaErrorContextIsDestroyed.code, cudaErrorContextIsDestroyed);
      put(cudaErrorAssert.code, cudaErrorAssert);
      put(cudaErrorTooManyPeers.code, cudaErrorTooManyPeers);
      put(cudaErrorHostMemoryAlreadyRegistered.code, cudaErrorHostMemoryAlreadyRegistered);
      put(cudaErrorHostMemoryNotRegistered.code, cudaErrorHostMemoryNotRegistered);
      put(cudaErrorHardwareStackError.code, cudaErrorHardwareStackError);
      put(cudaErrorIllegalInstruction.code, cudaErrorIllegalInstruction);
      put(cudaErrorMisalignedAddress.code, cudaErrorMisalignedAddress);
      put(cudaErrorInvalidAddressSpace.code, cudaErrorInvalidAddressSpace);
      put(cudaErrorInvalidPc.code, cudaErrorInvalidPc);
      put(cudaErrorLaunchFailure.code, cudaErrorLaunchFailure);
      put(cudaErrorCooperativeLaunchTooLarge.code, cudaErrorCooperativeLaunchTooLarge);
      put(cudaErrorNotPermitted.code, cudaErrorNotPermitted);
      put(cudaErrorNotSupported.code, cudaErrorNotSupported);
      put(cudaErrorSystemNotReady.code, cudaErrorSystemNotReady);
      put(cudaErrorSystemDriverMismatch.code, cudaErrorSystemDriverMismatch);
      put(cudaErrorCompatNotSupportedOnDevice.code, cudaErrorCompatNotSupportedOnDevice);
      put(cudaErrorMpsConnectionFailed.code, cudaErrorMpsConnectionFailed);
      put(cudaErrorMpsRpcFailure.code, cudaErrorMpsRpcFailure);
      put(cudaErrorMpsServerNotReady.code, cudaErrorMpsServerNotReady);
      put(cudaErrorMpsMaxClientsReached.code, cudaErrorMpsMaxClientsReached);
      put(cudaErrorMpsMaxConnectionsReached.code, cudaErrorMpsMaxConnectionsReached);
      put(cudaErrorStreamCaptureUnsupported.code, cudaErrorStreamCaptureUnsupported);
      put(cudaErrorStreamCaptureInvalidated.code, cudaErrorStreamCaptureInvalidated);
      put(cudaErrorStreamCaptureMerge.code, cudaErrorStreamCaptureMerge);
      put(cudaErrorStreamCaptureUnmatched.code, cudaErrorStreamCaptureUnmatched);
      put(cudaErrorStreamCaptureUnjoined.code, cudaErrorStreamCaptureUnjoined);
      put(cudaErrorStreamCaptureIsolation.code, cudaErrorStreamCaptureIsolation);
      put(cudaErrorStreamCaptureImplicit.code, cudaErrorStreamCaptureImplicit);
      put(cudaErrorCapturedEvent.code, cudaErrorCapturedEvent);
      put(cudaErrorStreamCaptureWrongThread.code, cudaErrorStreamCaptureWrongThread);
      put(cudaErrorTimeout.code, cudaErrorTimeout);
      put(cudaErrorGraphExecUpdateFailure.code, cudaErrorGraphExecUpdateFailure);
      put(cudaErrorExternalDevice.code, cudaErrorExternalDevice);
      put(cudaErrorUnknown.code, cudaErrorUnknown);
      put(cudaErrorApiFailureBase.code, cudaErrorApiFailureBase);
    }};

    CudaError(int errorCode) {
      this.code = errorCode;
    }

    public static CudaError parseErrorCode(int errorCode) {
      if (!codeToError.containsKey(errorCode)) {
        return UnknownNativeError;
      }
      return codeToError.get(errorCode);
    }

  }
}
