# Copyright (c) 2022, NVIDIA CORPORATION.
from numba import cuda

try:
    # allow import on CPU-only machine
    driver_maj, driver_min = cuda.cudadrv.driver.get_version()
    runtime_maj, runtime_min = cuda.cudadrv.runtime.runtime.get_version()

    if driver_maj >= runtime_maj and driver_min >= runtime_min:
        from strings_udf import lowering
        from pathlib import Path

        here = str(Path(__file__).parent.absolute())
        relative = "/../cpp/build/CMakeFiles/shim.dir/src/strings/udf/shim.ptx"
        ptxpath = here + relative
    else:
        raise NotImplementedError(
            "String UDFs require CUDA driver version >= CUDA runtime version"
        )

except cuda.cudadrv.driver.CudaAPIError:
    # no GPU found
    pass
