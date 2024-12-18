# Copyright (c) 2020-2024, NVIDIA CORPORATION.


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

    from cuda.bindings.runtime import cudaDeviceAttr, cudaError_t

    from rmm._cuda.gpu import (
        CUDARuntimeError,
        deviceGetName,
        driverGetVersion,
        getDeviceAttribute,
        getDeviceCount,
        runtimeGetVersion,
    )

    from cudf.errors import UnsupportedCUDAError

    notify_caller_errors = {
        cudaError_t.cudaErrorInitializationError,
        cudaError_t.cudaErrorInsufficientDriver,
        cudaError_t.cudaErrorInvalidDeviceFunction,
        cudaError_t.cudaErrorInvalidDevice,
        cudaError_t.cudaErrorStartupFailure,
        cudaError_t.cudaErrorInvalidKernelImage,
        cudaError_t.cudaErrorAlreadyAcquired,
        cudaError_t.cudaErrorOperatingSystem,
        cudaError_t.cudaErrorNotPermitted,
        cudaError_t.cudaErrorNotSupported,
        cudaError_t.cudaErrorSystemNotReady,
        cudaError_t.cudaErrorSystemDriverMismatch,
        cudaError_t.cudaErrorCompatNotSupportedOnDevice,
        cudaError_t.cudaErrorDeviceUninitialized,
        cudaError_t.cudaErrorTimeout,
        cudaError_t.cudaErrorUnknown,
        cudaError_t.cudaErrorApiFailureBase,
    }

    try:
        gpus_count = getDeviceCount()
    except CUDARuntimeError as e:
        if e.status in notify_caller_errors:
            raise e
        # If there is no GPU detected, set `gpus_count` to -1
        gpus_count = -1
    except RuntimeError as e:
        # getDeviceCount() can raise a RuntimeError
        # when ``libcuda.so`` is missing.
        # We don't want this to propagate up to the user.
        warnings.warn(str(e))
        return

    if gpus_count > 0:
        # Cupy throws RunTimeException to get GPU count,
        # hence obtaining GPU count by in-house cpp api above

        major_version = getDeviceAttribute(
            cudaDeviceAttr.cudaDevAttrComputeCapabilityMajor, 0
        )

        if major_version < 7:
            # A GPU with NVIDIA Volta™ architecture or newer is required.
            # Reference: https://developer.nvidia.com/cuda-gpus
            # Hardware Generation       Compute Capability
            #    Hopper                 9.x
            #    Ampere                 8.x
            #    Turing                 7.5
            #    Volta                  7.0, 7.2
            #    Pascal                 6.x
            #    Maxwell                5.x
            #    Kepler                 3.x
            #    Fermi                  2.x
            device_name = deviceGetName(0)
            minor_version = getDeviceAttribute(
                cudaDeviceAttr.cudaDevAttrComputeCapabilityMinor, 0
            )
            raise UnsupportedCUDAError(
                "A GPU with NVIDIA Volta™ (Compute Capability 7.0) "
                "or newer architecture is required.\n"
                f"Detected GPU 0: {device_name}\n"
                f"Detected Compute Capability: {major_version}.{minor_version}"
            )

        cuda_runtime_version = runtimeGetVersion()

        if cuda_runtime_version < 11000:
            # Require CUDA Runtime version 11.0 or greater.
            major_version = cuda_runtime_version // 1000
            minor_version = (cuda_runtime_version % 1000) // 10
            raise UnsupportedCUDAError(
                "Detected CUDA Runtime version is "
                f"{major_version}.{minor_version}. "
                "Please update your CUDA Runtime to 11.0 or above."
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
            raise UnsupportedCUDAError(
                "We couldn't detect the GPU driver properly. Please follow "
                "the installation guide to ensure your driver is properly "
                "installed: "
                "https://docs.nvidia.com/cuda/cuda-installation-guide-linux/"
            )
        elif cuda_driver_supported_rt_version >= cuda_runtime_version:
            # CUDA Driver Version Check:
            # Driver Runtime version is >= Runtime version
            pass
        elif (
            cuda_driver_supported_rt_version >= 11000
            and cuda_runtime_version >= 11000
        ):
            # With cuda enhanced compatibility any code compiled
            # with 11.x version of cuda can now run on any
            # driver >= 450.80.02. 11000 is the minimum cuda
            # version 450.80.02 supports.
            pass
        else:
            raise UnsupportedCUDAError(
                "Please update your NVIDIA GPU Driver to support CUDA "
                "Runtime.\n"
                f"Detected CUDA Runtime version : {cuda_runtime_version}\n"
                "Latest version of CUDA supported by current "
                f"NVIDIA GPU Driver : {cuda_driver_supported_rt_version}"
            )
    else:
        warnings.warn("No NVIDIA GPU detected")
