/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package ai.rapids.cudf;

import java.util.HashSet;
import java.util.Set;

/**
 * Exception from the cuda language/library.  Be aware that because of how cuda does asynchronous
 * processing exceptions from cuda can be thrown by method calls that did not cause the exception
 * to take place.  These will take place on the same thread that caused the error.
 * <p>
 * Please See
 * <a href="https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__ERROR.html">the cuda docs</a>
 * for more details on how this works.
 * <p>
 * In general you can recover from cuda errors even in async calls if you make sure that you
 * don't switch between threads for different parts of processing that can be retried as a chunk.
 */
public class CudaException extends RuntimeException {
  CudaException(String message) {
    super(message);
    this.cudaError = extractCudaError(message);
  }

  CudaException(String message, Throwable cause) {
    super(message, cause);
    this.cudaError = extractCudaError(message);
  }

  public final CudaError cudaError;

  /**
   * The Java mirror of cudaError, which facilities the tracking of CUDA errors in JVM.
   */
  public enum CudaError {
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

    private static final Set<CudaError> stickyErrors = new HashSet<CudaError>(){{
      add(CudaError.cudaErrorIllegalAddress);
      add(CudaError.cudaErrorLaunchTimeout);
      add(CudaError.cudaErrorHardwareStackError);
      add(CudaError.cudaErrorIllegalInstruction);
      add(CudaError.cudaErrorMisalignedAddress);
      add(CudaError.cudaErrorInvalidAddressSpace);
      add(CudaError.cudaErrorInvalidPc);
      add(CudaError.cudaErrorLaunchFailure);
      add(CudaError.cudaErrorExternalDevice);
      add(CudaError.cudaErrorUnknown);
    }};

    CudaError(int errorCode) {
      this.code = errorCode;
    }

    /**
     * Returns whether this CudaError is sticky or not.
     *
     * Sticky errors leave the process in an inconsistent state and any further CUDA work will return
     * the same error. To continue using CUDA, the process must be terminated and relaunched.
     */
    public boolean isSticky() {
      return stickyErrors.contains(this);
    }
  }

  private static CudaError extractCudaError(String message) {
    for (String segment : message.split(" ")) {
      if (segment.startsWith("cudaError")) {
        return CudaError.valueOf(segment);
      }
    }
    throw new CudfException("invalid CUDA error message: " + message);
  }
}
