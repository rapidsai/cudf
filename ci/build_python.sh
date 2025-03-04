#!/bin/bash
# Copyright (c) 2022-2025, NVIDIA CORPORATION.

set -euo pipefail

rapids-configure-conda-channels

source rapids-configure-sccache

source rapids-date-string

export CMAKE_GENERATOR=Ninja

rapids-print-env

rapids-generate-version > ./VERSION

rapids-logger "Begin py build"

CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)

sccache --zero-stats

# TODO: Remove `--no-test` flag once importing on a CPU
# node works correctly
# With boa installed conda build forwards to the boa builder

RAPIDS_PACKAGE_VERSION=$(head -1 ./VERSION) \
    rapids-telemetry-record build-pylibcudf.log \
        rapids-conda-retry build \
        --no-test \
        --channel "${CPP_CHANNEL}" \
        conda/recipes/pylibcudf

rapids-telemetry-record sccache-stats-pylibcudf.txt sccache --show-adv-stats
sccache --zero-stats

RAPIDS_PACKAGE_VERSION=$(head -1 ./VERSION) rapids-conda-retry build \
  --no-test \
  --channel "${CPP_CHANNEL}" \
  --channel "${RAPIDS_CONDA_BLD_OUTPUT_DIR}" \
  conda/recipes/cudf

sccache --show-adv-stats
sccache --zero-stats

RAPIDS_PACKAGE_VERSION=$(head -1 ./VERSION) \
    rapids-telemetry-record build-dask-cudf.log \
        rapids-conda-retry build \
        --no-test \
        --channel "${CPP_CHANNEL}" \
        --channel "${RAPIDS_CONDA_BLD_OUTPUT_DIR}" \
        conda/recipes/dask-cudf

RAPIDS_PACKAGE_VERSION=$(head -1 ./VERSION) \
    rapids-telemetry-record build-cudf_kafka.log \
        rapids-conda-retry build \
        --no-test \
        --channel "${CPP_CHANNEL}" \
        --channel "${RAPIDS_CONDA_BLD_OUTPUT_DIR}" \
        conda/recipes/cudf_kafka

rapids-telemetry-record sccache-stats-cudf_kafka.txt sccache --show-adv-stats
sccache --zero-stats

RAPIDS_PACKAGE_VERSION=$(head -1 ./VERSION) \
    rapids-telemetry-record build-custreamz.log \
        rapids-conda-retry build \
        --no-test \
        --channel "${CPP_CHANNEL}" \
  --channel "${RAPIDS_CONDA_BLD_OUTPUT_DIR}" \
        conda/recipes/custreamz

rapids-telemetry-record sccache-stats-custreamz.txt sccache --show-adv-stats
sccache --zero-stats

RAPIDS_PACKAGE_VERSION=$(head -1 ./VERSION) \
    rapids-telemetry-record build-cudf-polars.log \
        rapids-conda-retry build \
        --no-test \
        --channel "${CPP_CHANNEL}" \
        --channel "${RAPIDS_CONDA_BLD_OUTPUT_DIR}" \
        conda/recipes/cudf-polars

rapids-telemetry-record sccache-stats-cudf-polars.txt sccache --show-adv-stats
rapids-upload-conda-to-s3 python
