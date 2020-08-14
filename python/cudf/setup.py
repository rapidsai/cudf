# Copyright (c) 2018-2020, NVIDIA CORPORATION.
import ctypes
import os
import shutil
import sys
import sysconfig
from distutils.sysconfig import get_python_lib

import numpy as np
import pyarrow as pa
from Cython.Build import cythonize
from setuptools import find_packages, setup
from setuptools.extension import Extension

import versioneer


def cuda_detect():
    """Attempt to detect the version of CUDA present in the operating system.
    On Windows and Linux, the CUDA library is installed by the NVIDIA
    driver package, and is typically found in the standard library path,
    rather than with the CUDA SDK (which is optional for running CUDA apps).
    On macOS, the CUDA library is only installed with the CUDA SDK, and
    might not be in the library path.
    Returns: version string (Ex: '9.2') or None if CUDA not found.
    """

    system = sys.platform
    if system == "darwin":
        lib_filenames = [
            "libcuda.dylib",  # check library path first
            os.path.join(CUDA_HOME, "/lib/libcuda.dylib"),
        ]
    elif system == "linux":
        lib_filenames = [
            "libcuda.so",  # check library path first
            "/usr/lib64/nvidia/libcuda.so",  # Redhat/CentOS/Fedora
            "/usr/lib/x86_64-linux-gnu/libcuda.so",  # Ubuntu
        ]
    elif system == "win32":
        lib_filenames = ["nvcuda.dll"]
    else:
        return None  # CUDA not available for other operating systems

    if system == "wind32":
        dll = ctypes.windll
    else:
        dll = ctypes.cdll
    libcuda = None
    for lib_filename in lib_filenames:
        try:
            libcuda = dll.LoadLibrary(lib_filename)
            break
        except Exception:
            pass
    if libcuda is None:
        return None

    # Get CUDA version
    try:
        cuInit = libcuda.cuInit
        flags = ctypes.c_uint(0)
        ret = cuInit(flags)
        if ret != 0:
            return None

        cuDriverGetVersion = libcuda.cuDriverGetVersion
        version_int = ctypes.c_int(0)
        ret = cuDriverGetVersion(ctypes.byref(version_int))
        if ret != 0:
            return None

        # Convert version integer to version string
        value = version_int.value
        return "%d.%d" % (value // 1000, (value % 1000) // 10)
    except Exception:
        return None


install_requires = ["numba", "cython"]

cython_files = ["cudf/**/*.pyx"]

CUDA_HOME = os.environ.get("CUDA_HOME", False)
if not CUDA_HOME:
    path_to_cuda_gdb = shutil.which("cuda-gdb")
    if path_to_cuda_gdb is None:
        raise OSError(
            "Could not locate CUDA. "
            "Please set the environment variable "
            "CUDA_HOME to the path to the CUDA installation "
            "and try again."
        )
    CUDA_HOME = os.path.dirname(os.path.dirname(path_to_cuda_gdb))

if not os.path.isdir(CUDA_HOME):
    raise OSError(f"Invalid CUDA_HOME: directory does not exist: {CUDA_HOME}")

cuda_include_dir = os.path.join(CUDA_HOME, "include")

CUDF_ROOT = os.environ.get("CUDF_ROOT", "../../cpp/build/")

print("CUDA_VERSION", cuda_detect())
print("CUDA_HOME", CUDA_HOME)

try:
    nthreads = int(os.environ.get("PARALLEL_LEVEL", "0") or "0")
except Exception:
    nthreads = 0

extensions = [
    Extension(
        "*",
        sources=cython_files,
        include_dirs=[
            "../../cpp/include/cudf",
            "../../cpp/include",
            os.path.join(CUDF_ROOT, "include"),
            os.path.join(CUDF_ROOT, "_deps/libcudacxx-src/include"),
            os.path.join(
                os.path.dirname(sysconfig.get_path("include")),
                "libcudf/libcudacxx",
            ),
            os.path.dirname(sysconfig.get_path("include")),
            np.get_include(),
            pa.get_include(),
            cuda_include_dir,
        ],
        library_dirs=(
            pa.get_library_dirs()
            + [get_python_lib(), os.path.join(os.sys.prefix, "lib")]
        ),
        libraries=["cudf"] + pa.get_libraries(),
        language="c++",
        extra_compile_args=["-std=c++14"],
    )
]

setup(
    name="cudf",
    version=versioneer.get_version(),
    description="cuDF - GPU Dataframe",
    url="https://github.com/rapidsai/cudf",
    author="NVIDIA Corporation",
    license="Apache 2.0",
    classifiers=[
        "Intended Audience :: Developers",
        "Topic :: Database",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    # Include the separately-compiled shared library
    setup_requires=["cython"],
    ext_modules=cythonize(
        extensions,
        nthreads=nthreads,
        compiler_directives=dict(
            profile=False, language_level=3, embedsignature=True
        ),
        compile_time_env={"CUDA_VERSION": cuda_detect()},
    ),
    packages=find_packages(include=["cudf", "cudf.*"]),
    package_data=dict.fromkeys(
        find_packages(include=["cudf._lib*", "cudf._cuda*"]), ["*.pxd"],
    ),
    cmdclass=versioneer.get_cmdclass(),
    install_requires=install_requires,
    zip_safe=False,
)
