# Copyright (c) 2019-2023, NVIDIA CORPORATION.

from setuptools import find_packages, setup

setup(
    include_package_data=True,
    packages=find_packages(exclude=["tests", "tests.*"]),
    entry_points={
        "dask.dataframe.backends": [
            "cudf = dask_cudf.backends:CudfBackendEntrypoint",
        ]
    },
    zip_safe=False,
)
