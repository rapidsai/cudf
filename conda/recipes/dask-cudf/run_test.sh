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

# Dask & Distributed option to install main(nightly) or `conda-forge` packages.
export INSTALL_DASK_MAIN=0

# Dask version to install when `INSTALL_DASK_MAIN=0`
export DASK_STABLE_VERSION="2023.3.2"

# Install the conda-forge or nightly version of dask and distributed
if [[ "${INSTALL_DASK_MAIN}" == 1 ]]; then
    rapids-logger "rapids-mamba-retry install -c dask/label/dev 'dask/label/dev::dask' 'dask/label/dev::distributed'"
    rapids-mamba-retry install -c dask/label/dev "dask/label/dev::dask" "dask/label/dev::distributed"
else
    rapids-logger "rapids-mamba-retry install conda-forge::dask=={$DASK_STABLE_VERSION} conda-forge::distributed==2023.3.2.1 conda-forge::dask-core==2023.3.2 --force-reinstall"
    rapids-mamba-retry install conda-forge::dask=={$DASK_STABLE_VERSION} conda-forge::distributed=="2023.3.2.1" conda-forge::dask-core=="2023.3.2" --force-reinstall
fi

logger "python -c 'import dask_cudf'"
python -c "import dask_cudf"
