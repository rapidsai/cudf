# Copyright (c) 2019-2022, NVIDIA CORPORATION.

import os

import versioneer
from setuptools import find_packages, setup

install_requires = [
    "dask==2022.9.2",
    "distributed==2022.9.2",
    "fsspec>=0.6.0",
    "numpy",
    "pandas>=1.0,<1.6.0dev0",
    f"cudf{os.getenv('RAPIDS_PY_WHEEL_CUDA_SUFFIX', default='')}",
    "cupy-cuda11x",
]

extras_require = {
    "test": [
        "numpy",
        "pandas>=1.0,<1.6.0dev0",
        "pytest",
        "numba>=0.54",
    ]
}

setup(
    name="dask-cudf" + os.getenv("RAPIDS_PY_WHEEL_CUDA_SUFFIX", default=""),
    version=os.getenv(
        "RAPIDS_PY_WHEEL_VERSIONEER_OVERRIDE", default=versioneer.get_version()
    ),
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
    ],
    packages=find_packages(exclude=["tests", "tests.*"]),
    cmdclass=versioneer.get_cmdclass(),
    install_requires=install_requires,
    extras_require=extras_require,
)
