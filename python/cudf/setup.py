# Copyright (c) 2018-2022, NVIDIA CORPORATION.

import os
import re
import shutil
import subprocess
import sys
import sysconfig

# Must import in this order:
#   setuptools -> Cython.Distutils.build_ext -> setuptools.command.build_ext
# Otherwise, setuptools.command.build_ext ends up inheriting from
# Cython.Distutils.old_build_ext which we do not want
import setuptools

try:
    from Cython.Distutils.build_ext import new_build_ext as _build_ext
except ImportError:
    from setuptools.command.build_ext import build_ext as _build_ext

from distutils.spawn import find_executable
from distutils.sysconfig import get_python_lib

import numpy as np
import pyarrow as pa
import setuptools.command.build_ext
from setuptools import find_packages, setup
from setuptools.extension import Extension

import versioneer

install_requires = [
    "cachetools",
    "cuda-python>=11.5,<12.0",
    "fsspec>=0.6.0",
    "numba>=0.53.1",
    "numpy",
    "nvtx>=0.2.1",
    "packaging",
    "pandas>=1.0,<1.5.0dev0",
    "protobuf>=3.20.1,<3.21.0a0",
    "typing_extensions",
]

extras_require = {
    "test": [
        "pytest",
        "pytest-benchmark",
        "pytest-xdist",
        "hypothesis",
        "mimesis",
        "fastavro>=0.22.9",
        "python-snappy>=0.6.0",
        "pyorc",
        "msgpack",
        "transformers<=4.10.3",
    ]
}

cython_files = ["cudf/**/*.pyx"]


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


class build_ext_and_proto_no_debug(_build_ext):
    def build_extensions(self):
        def remove_flags(compiler, *flags):
            for flag in flags:
                try:
                    compiler.compiler_so = list(
                        filter((flag).__ne__, compiler.compiler_so)
                    )
                except Exception:
                    pass

        # Full optimization
        self.compiler.compiler_so.append("-O3")
        # Silence '-Wunknown-pragmas' warning
        self.compiler.compiler_so.append("-Wno-unknown-pragmas")
        # No debug symbols, full optimization, no '-Wstrict-prototypes' warning
        remove_flags(
            self.compiler, "-g", "-G", "-O1", "-O2", "-Wstrict-prototypes"
        )
        super().build_extensions()

    def finalize_options(self):
        if self.distribution.ext_modules:
            # Delay import this to allow for Cython-less installs
            from Cython.Build.Dependencies import cythonize

            nthreads = getattr(self, "parallel", None)  # -j option in Py3.5+
            nthreads = int(nthreads) if nthreads else None
            self.distribution.ext_modules = cythonize(
                self.distribution.ext_modules,
                nthreads=nthreads,
                force=self.force,
                gdb_debug=False,
                compiler_directives=dict(
                    profile=False, language_level=3, embedsignature=True
                ),
            )
        # Skip calling super() and jump straight to setuptools
        setuptools.command.build_ext.build_ext.finalize_options(self)

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
        _build_ext.run(self)


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
                "rapids/libcudacxx",
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
        extra_compile_args=["-std=c++17"],
    )
]

cmdclass = versioneer.get_cmdclass()
cmdclass["build_ext"] = build_ext_and_proto_no_debug


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
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    # Include the separately-compiled shared library
    setup_requires=["cython", "protobuf"],
    ext_modules=extensions,
    packages=find_packages(include=["cudf", "cudf.*"]),
    package_data=dict.fromkeys(
        find_packages(include=["cudf._lib*"]),
        ["*.pxd"],
    ),
    cmdclass=cmdclass,
    install_requires=install_requires,
    extras_require=extras_require,
    zip_safe=False,
)
