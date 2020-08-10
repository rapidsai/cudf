# Copyright (c) 2020, NVIDIA CORPORATION.

from setuptools import find_packages, setup

import versioneer

install_requires = ["cudf_kafka", "cudf"]

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
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    packages=find_packages(include=["custreamz", "custreamz.*"]),
    cmdclass=versioneer.get_cmdclass(),
    install_requires=install_requires,
    zip_safe=False,
)
