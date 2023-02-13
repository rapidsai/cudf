# Copyright (c) 2020-2023, NVIDIA CORPORATION.

import versioneer
from setuptools import find_packages, setup

install_requires = ["cudf_kafka", "cudf"]

extras_require = {"test": ["pytest", "pytest-xdist"]}

setup(
    name="custreamz",
    version=versioneer.get_version(),
    description="cuStreamz - GPU Accelerated Streaming",
    url="https://github.com/rapidsai/cudf",
    author="NVIDIA Corporation",
    license="Apache 2.0",
    classifiers=[
        "Intended Audience :: Developers",
        "Topic :: Streaming",
        "Topic :: Scientific/Engineering",
        "Topic :: Apache Kafka",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    packages=find_packages(include=["custreamz", "custreamz.*"]),
    cmdclass=versioneer.get_cmdclass(),
    install_requires=install_requires,
    extras_require=extras_require,
    zip_safe=False,
)
