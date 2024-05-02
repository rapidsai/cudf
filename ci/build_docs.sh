#!/bin/bash
# Copyright (c) 2023-2024, NVIDIA CORPORATION.

set -euo pipefail

export RAPIDS_VERSION="$(rapids-version)"
export RAPIDS_VERSION_MAJOR_MINOR="$(rapids-version-major-minor)"
export RAPIDS_VERSION_NUMBER="$RAPIDS_VERSION_MAJOR_MINOR"

rapids-logger "Create test conda environment"
. /opt/conda/etc/profile.d/conda.sh

ENV_YAML_DIR="$(mktemp -d)"

rapids-dependency-file-generator \
  --output conda \
  --file_key docs \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" | tee "${ENV_YAML_DIR}/env.yaml"

rapids-mamba-retry env create --yes -f "${ENV_YAML_DIR}/env.yaml" -n docs
conda activate docs

rapids-print-env

rapids-logger "Downloading artifacts from previous jobs"
CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)
PYTHON_CHANNEL=$(rapids-download-conda-from-s3 python)

rapids-mamba-retry install \
  --channel "${CPP_CHANNEL}" \
  --channel "${PYTHON_CHANNEL}" \
  libcudf cudf dask-cudf

export RAPIDS_DOCS_DIR="$(mktemp -d)"

rapids-logger "Build CPP docs"
pushd cpp/doxygen
aws s3 cp s3://rapidsai-docs/librmm/html/${RAPIDS_VERSION_NUMBER}/rmm.tag . || echo "Failed to download rmm Doxygen tag"
doxygen Doxyfile
mkdir -p "${RAPIDS_DOCS_DIR}/libcudf/html"
mv html/* "${RAPIDS_DOCS_DIR}/libcudf/html"
popd

rapids-logger "Build Python docs"
pushd docs/cudf
make dirhtml
mkdir -p "${RAPIDS_DOCS_DIR}/cudf/html"
mv build/dirhtml/* "${RAPIDS_DOCS_DIR}/cudf/html"
if [[ "${RAPIDS_BUILD_TYPE}" != "pull-request" ]]; then
  make text
  mkdir -p "${RAPIDS_DOCS_DIR}/cudf/txt"
  mv build/text/* "${RAPIDS_DOCS_DIR}/cudf/txt"
fi
popd

rapids-logger "Build dask-cuDF Sphinx docs"
pushd docs/dask_cudf
make dirhtml
mkdir -p "${RAPIDS_DOCS_DIR}/dask-cudf/html"
mv build/dirhtml/* "${RAPIDS_DOCS_DIR}/dask-cudf/html"
if [[ "${RAPIDS_BUILD_TYPE}" != "pull-request" ]]; then
  make text
  mkdir -p "${RAPIDS_DOCS_DIR}/dask-cudf/txt"
  mv build/text/* "${RAPIDS_DOCS_DIR}/dask-cudf/txt"
fi
popd

rapids-upload-docs
