#!/bin/bash
# Copyright (c) 2022-2025, NVIDIA CORPORATION.

set -euo pipefail

source rapids-configure-sccache

source rapids-date-string

export CMAKE_GENERATOR=Ninja

rapids-print-env

rapids-generate-version > ./VERSION

rapids-logger "Begin py build"

CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)

sccache --zero-stats


RAPIDS_PACKAGE_VERSION=$(head -1 ./VERSION)
export RAPIDS_PACKAGE_VERSION

# populates `RATTLER_CHANNELS` array
source rapids-rattler-channel-string

# TODO: Remove `--test skip` flag once importing on a CPU node works correctly

# --no-build-id allows for caching with `sccache`
# more info is available at
# https://rattler.build/latest/tips_and_tricks/#using-sccache-or-ccache-with-rattler-build
rattler-build build --recipe conda/recipes/pylibcudf \
                    --experimental \
                    --no-build-id \
                    --channel-priority disabled \
                    --output-dir "$RAPIDS_CONDA_BLD_OUTPUT_DIR" \
                    --test skip \
                    --channel "${CPP_CHANNEL}" \
                    "${RATTLER_CHANNELS[@]}"

sccache --show-adv-stats
sccache --zero-stats

rattler-build build --recipe conda/recipes/cudf \
                    --experimental \
                    --no-build-id \
                    --channel-priority disabled \
                    --output-dir "$RAPIDS_CONDA_BLD_OUTPUT_DIR" \
                    --test skip \
                    --channel "${CPP_CHANNEL}" \
                    "${RATTLER_CHANNELS[@]}"

sccache --show-adv-stats
sccache --zero-stats

rattler-build build --recipe conda/recipes/dask-cudf \
                    --experimental \
                    --no-build-id \
                    --channel-priority disabled \
                    --output-dir "$RAPIDS_CONDA_BLD_OUTPUT_DIR" \
                    --test skip \
                    --channel "${CPP_CHANNEL}" \
                    "${RATTLER_CHANNELS[@]}"

sccache --show-adv-stats
sccache --zero-stats

rattler-build build --recipe conda/recipes/cudf_kafka \
                    --experimental \
                    --no-build-id \
                    --channel-priority disabled \
                    --output-dir "$RAPIDS_CONDA_BLD_OUTPUT_DIR" \
                    --test skip \
                    --channel "${CPP_CHANNEL}" \
                    "${RATTLER_CHANNELS[@]}"

sccache --show-adv-stats
sccache --zero-stats

rattler-build build --recipe conda/recipes/custreamz \
                    --experimental \
                    --no-build-id \
                    --channel-priority disabled \
                    --output-dir "$RAPIDS_CONDA_BLD_OUTPUT_DIR" \
                    --test skip \
                    --channel "${CPP_CHANNEL}" \
                    "${RATTLER_CHANNELS[@]}"

sccache --show-adv-stats
sccache --zero-stats

rattler-build build --recipe conda/recipes/cudf-polars \
                    --experimental \
                    --no-build-id \
                    --channel-priority disabled \
                    --output-dir "$RAPIDS_CONDA_BLD_OUTPUT_DIR" \
                    --test skip \
                    --channel "${CPP_CHANNEL}" \
                    "${RATTLER_CHANNELS[@]}"

sccache --show-adv-stats

# remove build_cache directory
rm -rf "$RAPIDS_CONDA_BLD_OUTPUT_DIR"/build_cache

rapids-upload-conda-to-s3 python
