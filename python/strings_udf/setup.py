# Copyright (c) 2021-2022, NVIDIA CORPORATION.

import os
from distutils.sysconfig import get_python_lib

from Cython.Build import cythonize
from setuptools import find_packages
from skbuild import setup
from setuptools.extension import Extension

CUDA_HOME = "/usr/local/cuda"
cuda_include_dir = os.path.join(CUDA_HOME, "include")
cuda_lib_dir = os.path.join(CUDA_HOME, "lib64")

CONDA_PREFIX = os.environ.get("CONDA_PREFIX")
print(CONDA_PREFIX)
extensions = [
    Extension(
        "*",
        sources=["strings_udf/_lib/*.pyx"],
        include_dirs=[
            os.path.abspath(os.path.join(CONDA_PREFIX, "include/cudf")),
            os.path.join(CONDA_PREFIX, "include/rapids/libcudacxx"),
            "./cpp/include",
            cuda_include_dir,
        ],
        library_dirs=(
            [
                get_python_lib(),
                os.path.join(os.sys.prefix, "lib"),
                cuda_lib_dir,
                "cpp/build",
            ]
        ),
        libraries=["cudart", "cudf", "nvrtc", "cudf_strings_udf"],
        language="c++",
        extra_compile_args=["-std=c++17"],
    )
]

directives = dict(profile=False, language_level=3, embedsignature=True)

setup(
    name="strings_udf",
    description="cudf strings udf library",
    author="NVIDIA Corporation",
    setup_requires=["cython"],
    ext_modules=cythonize(extensions, compiler_directives=directives),
    zip_safe=False,
    packages=find_packages(),
)
