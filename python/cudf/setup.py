# Copyright (c) 2018-2022, NVIDIA CORPORATION.

import os
import re
import shutil
import subprocess
import sys
import platform
from distutils.spawn import find_executable

import versioneer
from setuptools import find_packages
from skbuild import setup
from skbuild.command.build_ext import build_ext


def get_cuda_version_from_header(cuda_include_dir, delimeter=""):

    cuda_version = None

    with open(os.path.join(cuda_include_dir, "cuda.h"), encoding="utf-8") as f:
        for line in f.readlines():
            if re.search(r"#define CUDA_VERSION ", line) is not None:
                cuda_version = line
                break

    if cuda_version is None:
        raise TypeError("CUDA_VERSION not found in cuda.h")
    cuda_version = int(cuda_version.split()[2])
    return "%d%s%d" % (
        cuda_version // 1000,
        delimeter,
        (cuda_version % 1000) // 10,
    )


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

cmdclass = versioneer.get_cmdclass()

cmake_args=[]

if os.getenv("CUDF_BUILD_WHEELS", "") != "":
    import pyarrow as pa
    cmake_args=[
        "-DCUDF_BUILD_WHEELS=ON",
        f"-DCUDF_PYARROW_WHEEL_DIR={pa.__path__[0]}",
    ]

cupy_requirement = []

if platform.machine() == "x86_64":
    cupy_requirement = [
        f"cupy-cuda{get_cuda_version_from_header(cuda_include_dir)}>=9.5.0,<12.0.0a0"
    ]

setup(
    name="cudf"+os.getenv("PYTHON_PACKAGE_CUDA_SUFFIX", default=""),
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
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    cmdclass=cmdclass,
    cmake_args=cmake_args,
    include_package_data=True,
    packages=find_packages(include=["cudf", "cudf.*"]),
    package_data={
        key: ["*.pxd"] for key in find_packages(include=["cudf._lib*"])
    },
    setup_requires=[
        f"rmm{os.getenv('PYTHON_PACKAGE_CUDA_SUFFIX', default='')}",
    ],
    install_requires=[
        "cachetools",
        "cuda-python>=11.5,<11.7.1",
        "fsspec>=0.6.0",
        "numba>=0.54",
        "numpy<1.23",
        "nvtx>=0.2.1",
        "packaging",
        "pandas>=1.0,<1.5.0dev0",
        "protobuf>=3.20.1,<3.21.0a0",
        "typing_extensions",
        "pyarrow==9.0.0",
        f"rmm{os.getenv('PYTHON_PACKAGE_CUDA_SUFFIX', default='')}",
    ] + cupy_requirement,
    extras_require={
        "test": [
            "pytest",
            "pytest-benchmark",
            "pytest-xdist",
            "hypothesis",
            "mimesis<4.1",
            "fastavro>=0.22.9",
            "python-snappy>=0.6.0",
            "pyorc",
            "msgpack",
            "transformers<=4.10.3",
            "tzdata",
        ]
    },
    zip_safe=False,
)
