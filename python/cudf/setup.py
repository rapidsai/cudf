# Copyright (c) 2018-2022, NVIDIA CORPORATION.

import os

import versioneer
from setuptools import find_packages
from skbuild import setup

install_requires = [
    "cachetools",
    "cuda-python>=11.7.1,<12.0",
    "fsspec>=0.6.0",
    "numba>=0.54",
    "numpy",
    "nvtx>=0.2.1",
    "packaging",
    "pandas>=1.0,<1.6.0dev0",
    "protobuf>=3.20.1,<3.21.0a0",
    "typing_extensions",
    "pyarrow==9.0.0",
    f"rmm{os.getenv('PYTHON_PACKAGE_CUDA_SUFFIX', default='')}",
    f"ptxcompiler{os.getenv('PYTHON_PACKAGE_CUDA_SUFFIX', default='')}",
    f"cubinlinker{os.getenv('PYTHON_PACKAGE_CUDA_SUFFIX', default='')}",
    # We may need to account for other architectures eventually. PEP 508 does
    # not appear to support an 'in list' syntax
    # `platform_machine in ('arch1', 'arch2', ...)
    # so we'll need to use multiple `or` conditions to support that case.
    "cupy-cuda11x;'x86' in platform_machine",
    "cupy-cuda11x @ https://pip.cupy.dev/aarch64;'aarch' in platform_machine",
]

extras_require = {
    "test": [
        "pytest",
        "pytest-benchmark",
        "pytest-xdist",
        "hypothesis",
        "mimesis>=4.1.0",
        "fastavro>=0.22.9",
        "python-snappy>=0.6.0",
        "pyorc",
        "msgpack",
        "transformers<=4.10.3",
        "tzdata",
    ]
}


setup(
    name="cudf" + os.getenv("RAPIDS_PY_WHEEL_CUDA_SUFFIX", default=""),
    version=os.getenv(
        "RAPIDS_PY_WHEEL_VERSIONEER_OVERRIDE", default=versioneer.get_version()
    ),
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
    cmdclass=versioneer.get_cmdclass(),
    include_package_data=True,
    packages=find_packages(include=["cudf", "cudf.*"]),
    package_data={
        key: ["*.pxd"] for key in find_packages(include=["cudf._lib*"])
    },
    # TODO: We need this to be dynamic, so it doesn't work to put it into
    # pyproject.toml, but setup_requires is deprecated so we need to find a
    # better solution for this.
    setup_requires=[
        f"rmm{os.getenv('PYTHON_PACKAGE_CUDA_SUFFIX', default='')}",
    ],
    install_requires=install_requires,
    extras_require=extras_require,
    zip_safe=False,
)
