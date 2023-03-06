# Copyright (c) 2018-2023, NVIDIA CORPORATION.

from setuptools import find_packages
from skbuild import setup

setup(
    include_package_data=True,
    packages=find_packages(include=["cudf", "cudf.*"]),
    zip_safe=False,
)
