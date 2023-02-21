# Copyright (c) 2018-2023, NVIDIA CORPORATION.

from setuptools import find_packages
from skbuild import setup

install_requires = [
    "cachetools",
    "cuda-python>=11.7.1,<12.0",
    "fsspec>=0.6.0",
    "numba>=0.56.2",
    "numpy",
    "nvtx>=0.2.1",
    "packaging",
    "pandas>=1.0,<1.6.0dev0",
    "protobuf==4.21",
    "typing_extensions",
    # Allow floating minor versions for Arrow.
    "pyarrow==10",
    "rmm==23.4.*",
    "ptxcompiler",
    "cubinlinker",
    "cupy-cuda11x",
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
        "transformers==4.24.0",
        "tzdata",
    ]
}

setup(
    name="cudf",
    version="23.04.00",
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
        "Programming Language :: Python :: 3.10",
    ],
    include_package_data=True,
    packages=find_packages(include=["cudf", "cudf.*"]),
    package_data={
        key: ["*.pxd"] for key in find_packages(include=["cudf._lib*"])
    },
    install_requires=install_requires,
    extras_require=extras_require,
    zip_safe=False,
)
