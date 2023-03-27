# Copyright (c) 2018-2023, NVIDIA CORPORATION.

from setuptools import find_packages
from skbuild import setup

packages = find_packages(include=["cudf*", "udf_cpp*"])
setup(
    packages=packages,
    package_data={key: ["*.pxd", "*.hpp", "*.cuh"] for key in packages},
    zip_safe=False,
)
