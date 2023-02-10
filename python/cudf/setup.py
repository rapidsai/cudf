# Copyright (c) 2018-2023, NVIDIA CORPORATION.

import os

import versioneer
from setuptools import find_packages
from skbuild import setup

cuda_suffix = os.getenv("RAPIDS_PY_WHEEL_CUDA_SUFFIX", default="")

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
    f"rmm{cuda_suffix}==23.4.*",
    f"ptxcompiler{cuda_suffix}",
    f"cubinlinker{cuda_suffix}",
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

if "RAPIDS_PY_WHEEL_VERSIONEER_OVERRIDE" in os.environ:
    orig_get_versions = versioneer.get_versions

    version_override = os.environ["RAPIDS_PY_WHEEL_VERSIONEER_OVERRIDE"]

    def get_versions():
        data = orig_get_versions()
        data["version"] = version_override
        return data

    versioneer.get_versions = get_versions

setup(
    name=f"cudf{cuda_suffix}",
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
        "Programming Language :: Python :: 3.10",
    ],
    cmdclass=versioneer.get_cmdclass(),
    include_package_data=True,
    packages=find_packages(include=["cudf", "cudf.*"]),
    package_data={
        key: ["*.pxd"] for key in find_packages(include=["cudf._lib*"])
    },
    install_requires=install_requires,
    extras_require=extras_require,
    zip_safe=False,
)
