# Copyright (c) 2019-2023, NVIDIA CORPORATION.

import os

from setuptools import find_packages, setup

cuda_suffix = os.getenv("RAPIDS_PY_WHEEL_CUDA_SUFFIX", default="")

install_requires = [
    "dask>=2023.1.1",
    "distributed>=2023.1.1",
    "fsspec>=0.6.0",
    "numpy",
    "pandas>=1.0,<1.6.0dev0",
    f"cudf{cuda_suffix}==23.4.*",
    "cupy-cuda11x",
]

extras_require = {
    "test": [
        "numpy",
        "pandas>=1.0,<1.6.0dev0",
        "pytest",
        "pytest-xdist",
        "numba>=0.56.2",
    ]
}

setup(
    name=f"dask-cudf{cuda_suffix}",
    version="23.04.00",
    description="Utilities for Dask and cuDF interactions",
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
    packages=find_packages(exclude=["tests", "tests.*"]),
    install_requires=install_requires,
    extras_require=extras_require,
)
