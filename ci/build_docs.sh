#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION.

set -euo pipefail

rapids-logger "Create test conda environment"
. /opt/conda/etc/profile.d/conda.sh

rapids-dependency-file-generator \
  --output conda \
  --file_key docs \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" | tee env.yaml

rapids-mamba-retry env create --force -f env.yaml -n docs
conda activate docs

rapids-print-env

rapids-logger "Downloading artifacts from previous jobs"
CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)
PYTHON_CHANNEL=$(rapids-download-conda-from-s3 python)
VERSION_NUMBER="23.06"

rapids-mamba-retry install \
  --channel "${CPP_CHANNEL}" \
  --channel "${PYTHON_CHANNEL}" \
  libcudf cudf dask-cudf


rapids-logger "Build Doxygen docs"
pushd cpp/doxygen
aws s3 cp s3://rapidsai-docs/librmm/${VERSION_NUMBER}/html/rmm.tag . || echo "Failed to download rmm Doxygen tag"
doxygen Doxyfile
popd

rapids-logger "Build cuDF Sphinx docs"
pushd docs/cudf
sphinx-build -b dirhtml source _html
sphinx-build -b text source _text
popd


rapids-logger "Build dask-cuDF Sphinx docs"
pushd docs/dask_cudf
sphinx-build -b dirhtml source _html
sphinx-build -b text source _text
popd


if [[ "${RAPIDS_BUILD_TYPE}" != "pull-request" ]]; then
  rapids-logger "Upload Docs to S3"
  aws s3 sync --no-progress --delete cpp/doxygen/html "s3://rapidsai-docs/libcudf/${VERSION_NUMBER}/html"
  aws s3 sync --no-progress --delete docs/cudf/_html "s3://rapidsai-docs/cudf/${VERSION_NUMBER}/html"
  aws s3 sync --no-progress --delete docs/cudf/_text "s3://rapidsai-docs/cudf/${VERSION_NUMBER}/txt"
  aws s3 sync --no-progress --delete docs/dask_cudf/_html "s3://rapidsai-docs/dask-cudf/${VERSION_NUMBER}/html"
  aws s3 sync --no-progress --delete docs/dask_cudf/_text "s3://rapidsai-docs/dask-cudf/${VERSION_NUMBER}/txt"
fi
