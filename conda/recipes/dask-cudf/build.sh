# Copyright (c) 2018-2019, NVIDIA CORPORATION.

# Install the master version of dask, distributed, and streamz

pip install "git+https://github.com/dask/distributed.git" --upgrade --no-deps

pip install "git+https://github.com/dask/dask.git" --upgrade --no-deps

pip install "git+https://github.com/python-streamz/streamz.git" --upgrade --no-deps

# This assumes the script is executed from the root of the repo directory
./build.sh dask_cudf
