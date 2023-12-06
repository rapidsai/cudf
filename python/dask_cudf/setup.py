# Copyright (c) 2019-2023, NVIDIA CORPORATION.

from setuptools import find_packages, setup

packages = find_packages(exclude=["tests", "tests.*"])

setup(
    include_package_data=True,
    packages=packages,
    package_data={key: ["VERSION"] for key in packages},
    entry_points={
        "dask.dataframe.backends": [
            "cudf = dask_cudf.backends:CudfBackendEntrypoint",
        ]
    },
    zip_safe=False,
)
