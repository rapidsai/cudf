# Copyright (c) 2019-2023, NVIDIA CORPORATION.

import os

import versioneer
from setuptools import find_packages, setup

cuda_suffix = os.getenv("RAPIDS_PY_WHEEL_CUDA_SUFFIX", default="")

install_requires = [
    "dask>=2022.12.0",
    "distributed>=2022.12.0",
    "fsspec>=0.6.0",
    "numpy",
    "pandas>=1.0,<1.6.0dev0",
    f"cudf{cuda_suffix}",
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

if "RAPIDS_PY_WHEEL_VERSIONEER_OVERRIDE" in os.environ:
    orig_get_versions = versioneer.get_versions

    version_override = os.environ["RAPIDS_PY_WHEEL_VERSIONEER_OVERRIDE"]

    def get_versions():
        data = orig_get_versions()
        data["version"] = version_override
        return data

    versioneer.get_versions = get_versions

setup(
    name=f"dask-cudf{cuda_suffix}",
    version=versioneer.get_version(),
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
    cmdclass=versioneer.get_cmdclass(),
    install_requires=install_requires,
    extras_require=extras_require,
)
