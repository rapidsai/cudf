# Copyright (c) 2020-2023, NVIDIA CORPORATION.

from setuptools import find_packages, setup

setup(
    include_package_data=True,
    packages=find_packages(include=["custreamz", "custreamz.*"]),
)
