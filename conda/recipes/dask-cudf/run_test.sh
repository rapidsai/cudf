#!/bin/bash
# Copyright (c) 2020-2023, NVIDIA CORPORATION.

set -e

# Logger function for build status output
function logger() {
  echo -e "\n>>>> $@\n"
}

# Importing cudf on arm64 CPU only nodes is currently not working due to a
# difference in reported gpu devices between arm64 and amd64
ARCH=$(arch)

if [ "${ARCH}" = "aarch64" ]; then
  logger "Skipping tests on arm64"
  exit 0
fi

# Dask & Distributed option to install main or `DASK_STABLE_VERSION` packages.
export INSTALL_DASK_MAIN=0

# Dask version to install when `INSTALL_DASK_MAIN=0`
export DASK_STABLE_VERSION="2023.1.1"

# Install the latest(main branch) version of dask and distributed
if [[ "${INSTALL_DASK_MAIN}" == 1 ]]; then
    export DASK_STABLE_VERSION="main"
fi

logger "pip install git+https://github.com/dask/distributed.git@{$DASK_STABLE_VERSION} --upgrade --no-deps"
pip install "git+https://github.com/dask/distributed.git@{$DASK_STABLE_VERSION}" --upgrade --no-deps

logger "pip install git+https://github.com/dask/dask.git@{$DASK_STABLE_VERSION} --upgrade --no-deps"
pip install "git+https://github.com/dask/dask.git@{$DASK_STABLE_VERSION}" --upgrade --no-deps

logger "python -c 'import dask_cudf'"
python -c "import dask_cudf"
