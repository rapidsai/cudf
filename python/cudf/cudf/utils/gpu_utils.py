# Copyright (c) 2020, NVIDIA CORPORATION.


def validate_setup():
    import os

    # TODO: Remove the following check once we arrive at a solution for #4827
    # This is a temporary workaround to unblock internal testing
    # related issue: https://github.com/rapidsai/cudf/issues/4827
    if (
        "RAPIDS_NO_INITIALIZE" in os.environ
        or "CUDF_NO_INITIALIZE" in os.environ
    ):
        return

    import warnings

    from rmm._cuda.gpu import (
        CUDARuntimeError,
        cudaDeviceAttr,
        cudaError,
        deviceGetName,
        driverGetVersion,
        getDeviceAttribute,
        getDeviceCount,
        runtimeGetVersion,
    )

    def _try_get_old_or_new_symbols():
        try:
            # CUDA 10.2+ symbols
            return [
                cudaError.cudaErrorDeviceUninitialized,
                cudaError.cudaErrorTimeout,
            ]
        except AttributeError:
            # CUDA 10.1 symbols
            return [cudaError.cudaErrorDeviceUninitilialized]

    notify_caller_errors = {
        cudaError.cudaErrorInitializationError,
        cudaError.cudaErrorInsufficientDriver,
        cudaError.cudaErrorInvalidDeviceFunction,
        cudaError.cudaErrorInvalidDevice,
        cudaError.cudaErrorStartupFailure,
        cudaError.cudaErrorInvalidKernelImage,
        cudaError.cudaErrorAlreadyAcquired,
        cudaError.cudaErrorOperatingSystem,
        cudaError.cudaErrorNotPermitted,
        cudaError.cudaErrorNotSupported,
        cudaError.cudaErrorSystemNotReady,
        cudaError.cudaErrorSystemDriverMismatch,
        cudaError.cudaErrorCompatNotSupportedOnDevice,
        *_try_get_old_or_new_symbols(),
        cudaError.cudaErrorUnknown,
        cudaError.cudaErrorApiFailureBase,
    }

    try:
        gpus_count = getDeviceCount()
    except CUDARuntimeError as e:
        if e.status in notify_caller_errors:
            raise e
        # If there is no GPU detected, set `gpus_count` to -1
        gpus_count = -1

    if gpus_count > 0:
        # Cupy throws RunTimeException to get GPU count,
        # hence obtaining GPU count by in-house cpp api above

        # 75 - Indicates to get "cudaDevAttrComputeCapabilityMajor" attribute
        # 0 - Get GPU 0
        major_version = getDeviceAttribute(
            cudaDeviceAttr.cudaDevAttrComputeCapabilityMajor, 0
        )

        if major_version >= 6:
            # You have a GPU with NVIDIA Pascal™ architecture or better
            # Hardware Generation	Compute Capability
            #    Turing	                7.5
            #    Volta	                7.x
            #    Pascal	                6.x
            #    Maxwell	              5.x
            #    Kepler	                3.x
            #    Fermi	                2.x
            pass
        else:
            device_name = deviceGetName(0)
            minor_version = getDeviceAttribute(
                cudaDeviceAttr.cudaDevAttrComputeCapabilityMinor, 0
            )
            warnings.warn(
                f"You will need a GPU with NVIDIA Pascal™ or "
                f"newer architecture"
                f"\nDetected GPU 0: {device_name} \n"
                f"Detected Compute Capability: "
                f"{major_version}.{minor_version}"
            )

        cuda_runtime_version = runtimeGetVersion()

        if cuda_runtime_version >= 10000:
            # CUDA Runtime Version Check: Runtime version is greater than 10000
            pass
        else:
            from cudf.errors import UnSupportedCUDAError

            minor_version = cuda_runtime_version % 100
            major_version = (cuda_runtime_version - minor_version) // 1000
            raise UnSupportedCUDAError(
                f"Detected CUDA Runtime version is "
                f"{major_version}.{str(minor_version)[0]}"
                f"Please update your CUDA Runtime to 10.0 or above"
            )

        cuda_driver_supported_rt_version = driverGetVersion()

        # Though Yes, Externally driver version is represented like `418.39`
        # and cuda runtime version like `10.1`. It is not the similar case
        # at cuda api's level. Coming down to APIs they follow a uniform
        # convention of an integer which corresponds to the versioning
        # like (1000 major + 10 minor) for 10.1 Driver version API doesn't
        # actually indicate driver version, it indicates only the latest
        # CUDA version supported by the driver.
        # For reference :
        # https://docs.nvidia.com/deploy/cuda-compatibility/index.html

        if cuda_driver_supported_rt_version == 0:
            from cudf.errors import UnSupportedCUDAError

            raise UnSupportedCUDAError(
                "We couldn't detect the GPU driver "
                "properly. Please follow the linux installation guide to "
                "ensure your driver is properly installed "
                ": https://docs.nvidia.com/cuda/cuda-installation-guide-linux/"
            )

        elif cuda_driver_supported_rt_version >= cuda_runtime_version:
            # CUDA Driver Version Check:
            # Driver Runtime version is >= Runtime version
            pass
        else:
            from cudf.errors import UnSupportedCUDAError

            raise UnSupportedCUDAError(
                f"Please update your NVIDIA GPU Driver to support CUDA "
                f"Runtime.\n"
                f"Detected CUDA Runtime version : {cuda_runtime_version}"
                f"\n"
                f"Latest version of CUDA supported by current "
                f"NVIDIA GPU Driver : {cuda_driver_supported_rt_version}"
            )

    else:

        warnings.warn("No NVIDIA GPU detected")
