# Install the master version of dask, distributed, and streamz

pip install "git+https://github.com/dask/distributed.git" --upgrade --no-deps

pip install "git+https://github.com/dask/dask.git" --upgrade --no-deps

python -c "import dask_cudf"