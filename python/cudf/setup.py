# Copyright (c) 2018-2021, NVIDIA CORPORATION.

import os
import re
import shutil
import subprocess
import sys
import sysconfig
from distutils.spawn import find_executable
from distutils.sysconfig import get_python_lib

import numpy as np
import pyarrow as pa
from Cython.Build import cythonize
from Cython.Distutils import build_ext
from setuptools import find_packages, setup
from setuptools.extension import Extension

import versioneer

install_requires = [
    "numba>=0.49.0,!=0.51.0",
    "Cython>=0.29,<0.30",
    "fastavro>=0.22.9",
    "fsspec>=0.6.0",
    "numpy",
    "pandas>=1.0,<1.3.0dev0",
    "typing_extensions",
    "protobuf",
    "nvtx>=0.2.1",
    "cachetools",
    "packaging",
]

extras_require = {
    "test": [
        "pytest",
        "pytest-benchmark",
        "pytest-xdist",
        "hypothesis" "mimesis",
        "pyorc",
        "msgpack",
    ]
}

cython_files = ["cudf/**/*.pyx"]


def get_cuda_version_from_header(cuda_include_dir, delimeter=""):

    cuda_version = None

    with open(
        os.path.join(cuda_include_dir, "cuda.h"), "r", encoding="utf-8"
    ) as f:
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
cuda_lib_dir = os.path.join(CUDA_HOME, "lib64")
install_requires.append(
    "cupy-cuda" + get_cuda_version_from_header(cuda_include_dir)
)

CUDF_HOME = os.environ.get(
    "CUDF_HOME",
    os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../")
    ),
)
CUDF_ROOT = os.environ.get(
    "CUDF_ROOT",
    os.path.abspath(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "../../cpp/build/"
        )
    ),
)

try:
    nthreads = int(os.environ.get("PARALLEL_LEVEL", "0") or "0")
except Exception:
    nthreads = 0

cmdclass = versioneer.get_cmdclass()


class build_ext_and_proto(build_ext):
    def build_extensions(self):
        try:
            # Silence the '-Wstrict-prototypes' warning
            self.compiler.compiler_so.remove("-Wstrict-prototypes")
        except Exception:
            pass
        build_ext.build_extensions(self)

    def run(self):
        # Get protoc
        protoc = None
        if "PROTOC" in os.environ and os.path.exists(os.environ["PROTOC"]):
            protoc = os.environ["PROTOC"]
        else:
            protoc = find_executable("protoc")
        if protoc is None:
            sys.stderr.write("protoc not found")
            sys.exit(1)

        # Build .proto file
        for source in ["cudf/utils/metadata/orc_column_statistics.proto"]:
            output = source.replace(".proto", "_pb2.py")

            if not os.path.exists(output) or (
                os.path.getmtime(source) > os.path.getmtime(output)
            ):
                with open(output, "a") as src:
                    src.write("# flake8: noqa" + os.linesep)
                    src.write("# fmt: off" + os.linesep)
                subprocess.check_call([protoc, "--python_out=.", source])
                with open(output, "r+") as src:
                    new_src_content = (
                        "# flake8: noqa"
                        + os.linesep
                        + "# fmt: off"
                        + os.linesep
                        + src.read()
                        + "# fmt: on"
                        + os.linesep
                    )
                    src.seek(0)
                    src.write(new_src_content)

        # Run original Cython build_ext command
        build_ext.run(self)


cmdclass["build_ext"] = build_ext_and_proto

extensions = [
    Extension(
        "*",
        sources=cython_files,
        include_dirs=[
            os.path.abspath(os.path.join(CUDF_HOME, "cpp/include/cudf")),
            os.path.abspath(os.path.join(CUDF_HOME, "cpp/include")),
            os.path.abspath(os.path.join(CUDF_ROOT, "include")),
            os.path.join(CUDF_ROOT, "_deps/libcudacxx-src/include"),
            os.path.join(CUDF_ROOT, "_deps/dlpack-src/include"),
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
            + [
                get_python_lib(),
                os.path.join(os.sys.prefix, "lib"),
                cuda_lib_dir,
            ]
        ),
        libraries=["cudart", "cudf"] + pa.get_libraries() + ["arrow_cuda"],
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
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    # Include the separately-compiled shared library
    setup_requires=["cython", "protobuf"],
    ext_modules=cythonize(
        extensions,
        nthreads=nthreads,
        compiler_directives=dict(
            profile=False, language_level=3, embedsignature=True
        ),
    ),
    packages=find_packages(include=["cudf", "cudf.*"]),
    package_data=dict.fromkeys(
        find_packages(include=["cudf._lib*"]), ["*.pxd"],
    ),
    cmdclass=cmdclass,
    install_requires=install_requires,
    zip_safe=False,
    extras_require=extras_require,
)
