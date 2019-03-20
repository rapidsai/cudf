#!/bin/bash

set -e

# COPYRIGHT 2019 NVIDIA
# conda_prepare.sh creates and activates the correct conda environment for cudf
# or updates it based on the latest .yml file.

if ! type nvcc 2>/dev/null; then
  echo 'Please put nvcc on your path before executing this script'
  exit
fi

# If CUDF is installed then we need to update, otherwise we need to create
CUDF=`conda list | grep cudf`
if [[ $CUDF == *"cudf"* ]]; then
  conda env update -f conda/environments/cudf_dev.yml
else
  conda env create --name cudf_dev --file conda/environments/cudf_dev.yml
  source activate cudf_dev
fi

# NVStrings has CUDF dependency that can only be installed post environment
# creation
NVCC_VER=`nvcc --version`
PY_VER=`python --version`
if [[ $PY_VER == *"3.7"* ]]; then
  PY_VER='3.7'
else
  PY_VER='3.6'
fi
if [[ $NVCC_VER == *"9.2"* ]]; then
  conda install -c nvidia -c rapidsai -c numba -c conda-forge -c defaults \
      nvstrings=0.3 python=$PY_VER
else
  conda install -c nvidia/label/cuda10.0 -c rapidsai/label/cuda10.0 -c numba \
        -c conda-forge -c defaults nvstrings=0.3 python=$PY_VER
fi
