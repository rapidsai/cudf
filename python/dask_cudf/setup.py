# Copyright (c) 2019, NVIDIA CORPORATION.
from setuptools import find_packages, setup

import versioneer

install_requires = ["cudf", "dask", "distributed"]

setup(
    name="dask-cudf",
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
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    packages=find_packages(exclude=["tests", "tests.*"]),
    cmdclass=versioneer.get_cmdclass(),
    install_requires=install_requires,
)
