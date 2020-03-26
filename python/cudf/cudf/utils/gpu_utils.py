def validate_setup():
    from .gpu import get_gpu_device_count

    gpus_count = get_gpu_device_count()

    if gpus_count > 0:
        # Cupy throws RunTimeException to get GPU count,
        # hence obtaining GPU count by in-house cpp api above
        import cupy

        # 75 - Indicates to get "cudaDevAttrComputeCapabilityMajor" attribute
        # 0 - Get GPU 0
        major_version = cupy.cuda.runtime.deviceGetAttribute(75, 0)

        if major_version >= 6:
            # You have a GPU with NVIDIA Pascal™ architecture or better
            # Hardware Generation	Compute Capability
            #    Turing	                7.5
            #    Volta	                7.x
            #    Pascal	                6.x
            #    Maxwell	            5.x
            #    Kepler	                3.x
            #    Fermi	                2.x
            pass
        else:
            from cudf.errors import UnSupportedGPUError

            raise UnSupportedGPUError(
                "You will need a GPU with NVIDIA Pascal™ architecture or \
                    better"
            )

        cuda_runtime_version = cupy.cuda.runtime.runtimeGetVersion()

        if cuda_runtime_version > 10000:
            # CUDA Runtime Version Check: Runtime version is greater than 10000
            pass
        else:
            from cudf.errors import UnSupportedCUDAError

            raise UnSupportedCUDAError(
                "Please update your CUDA Runtime to 10.0 or above"
            )

        cuda_driver_version = cupy.cuda.runtime.driverGetVersion()

        if cuda_driver_version == 0:
            from cudf.errors import UnSupportedCUDAError

            raise UnSupportedCUDAError("Please install CUDA Driver")
        elif cuda_driver_version >= cuda_runtime_version:
            # CUDA Driver Version Check:
            # Driver Runtime version is >= Runtime version
            pass
        else:
            from cudf.errors import UnSupportedCUDAError

            raise UnSupportedCUDAError(
                "Please update your NVIDIA GPU Driver to support CUDA \
                    Runtime.\n"
                "Detected CUDA Runtime version : "
                + str(cuda_runtime_version)
                + "\n"
                "Latest version of CUDA \
                    supported by current NVIDIA GPU Driver : "
                + str(cuda_driver_version)
            )

    else:
        import warnings

        warnings.warn(
            "You donot have an NVIDIA GPU, please install one and try again"
        )
