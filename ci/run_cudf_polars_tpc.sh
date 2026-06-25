#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

source rapids-init-pip

rapids-logger "Download wheels"

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"
CUDF_POLARS_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="cudf_polars_${RAPIDS_PY_CUDA_SUFFIX}" RAPIDS_PY_WHEEL_PURE="1" rapids-download-wheels-from-github python)
LIBCUDF_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="libcudf_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github cpp)
PYLIBCUDF_WHEELHOUSE=$(rapids-download-from-github "$(rapids-package-name "wheel_python" pylibcudf --stable --cuda "$RAPIDS_CUDA_VERSION")")

rapids-generate-pip-constraints py_test_cudf_polars "${PIP_CONSTRAINT}"

rapids-logger "Installing cudf_polars and TPC test dependencies"

TPCH_REQUIREMENTS=$(mktemp --suffix=.txt)
rapids-dependency-file-generator \
    --config dependencies.yaml \
    --file-key test_cudf_polars_tpch \
    --output requirements \
    > "${TPCH_REQUIREMENTS}"

rapids-pip-retry install \
    -v \
    --prefer-binary \
    --constraint "${PIP_CONSTRAINT}" \
    "$(echo "${CUDF_POLARS_WHEELHOUSE}"/cudf_polars_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)[test]" \
    "$(echo "${LIBCUDF_WHEELHOUSE}"/libcudf_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)" \
    "$(echo "${PYLIBCUDF_WHEELHOUSE}"/pylibcudf_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)" \
    -r "${TPCH_REQUIREMENTS}"

rapids-logger "Generating TPC-H data at SF=0.01"

export TPCH_DATA_DIR
TPCH_DATA_DIR=$(mktemp -d)
tpchgen-cli -s 0.01 --format=parquet --parts=4 --output-dir="${TPCH_DATA_DIR}"

rapids-logger "Generating TPC-DS data at SF=0.01"

export TPCDS_DATA_DIR
TPCDS_DATA_DIR=$(mktemp -d)
python3 "$(dirname "$0")/generate_tpcds_data.py" --scale 0.01 --output-dir "${TPCDS_DATA_DIR}"

rapids-logger "Running TPC-H validation tests"

cd python/cudf_polars

python -m cudf_polars.streaming.benchmarks.pdsh all \
    --path "${TPCH_DATA_DIR}" \
    --suffix "/*.parquet" \
    --frontend spmd \
    --validate-against duckdb \
    --iterations 2

rapids-logger "Running TPC-DS validation tests"

python -m cudf_polars.streaming.benchmarks.pdsds all \
    --path "${TPCDS_DATA_DIR}" \
    --scale 0.01 \
    --qualification \
    --frontend spmd \
    --validate-against duckdb \
    --iterations 2
