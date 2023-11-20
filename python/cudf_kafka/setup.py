# Copyright (c) 2018-2023, NVIDIA CORPORATION.
from setuptools import find_packages
from skbuild import setup

packages = find_packages(include=["cudf_kafka*"])

setup(
    packages=packages,
    package_data={
        key: ["VERSION", "*.pxd", "*.hpp", "*.cuh"] for key in packages
    },
    zip_safe=False,
)
