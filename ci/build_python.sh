#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

source rapids-configure-sccache
source rapids-date-string

export CMAKE_GENERATOR=Ninja

rapids-print-env

rapids-generate-version > ./VERSION
rapids-generate-version > ./python/cudf/cudf/VERSION

rapids-logger "Begin py build"

CPP_CHANNEL=$(rapids-download-conda-from-github cpp)

RAPIDS_PACKAGE_VERSION=$(head -1 ./VERSION)
export RAPIDS_PACKAGE_VERSION

# populates `RATTLER_CHANNELS` array and `RATTLER_ARGS` array
source rapids-rattler-channel-string

rapids-logger "Prepending channel ${CPP_CHANNEL} to RATTLER_CHANNELS"

RATTLER_CHANNELS=("--channel" "${CPP_CHANNEL}" "${RATTLER_CHANNELS[@]}")

sccache --stop-server 2>/dev/null || true


# Only use stable ABI package naming for Python >= 3.11
if [[ "${RAPIDS_PY_VERSION}" != "3.10" ]]; then
  RECIPE_PATH="conda/recipes/pylibcudf"
else
  RECIPE_PATH="conda/recipes/pylibcudf_py310"
fi

rapids-logger "Building pylibcudf"

# --no-build-id allows for caching with `sccache`
# more info is available at
# https://rattler.build/latest/tips_and_tricks/#using-sccache-or-ccache-with-rattler-build
rapids-telemetry-record build-pylibcudf.log \
  rattler-build build --recipe "${RECIPE_PATH}" \
                    "${RATTLER_ARGS[@]}" \
                    "${RATTLER_CHANNELS[@]}"

rapids-telemetry-record sccache-stats-pylibcudf.txt sccache --show-adv-stats
sccache --stop-server >/dev/null 2>&1 || true


# Only use stable ABI package naming for Python >= 3.11
if [[ "${RAPIDS_PY_VERSION}" != "3.10" ]]; then
  RECIPE_PATH="conda/recipes/cudf"
else
  RECIPE_PATH="conda/recipes/cudf_py310"
fi

rapids-logger "Building cudf"

rapids-telemetry-record build-cudf.log \
   rattler-build build --recipe "${RECIPE_PATH}" \
                    "${RATTLER_ARGS[@]}" \
                    "${RATTLER_CHANNELS[@]}"

rapids-telemetry-record sccache-stats-cudf.txt sccache --show-adv-stats
sccache --stop-server >/dev/null 2>&1 || true

# Only use stable ABI package naming for Python >= 3.11
if [[ "${RAPIDS_PY_VERSION}" != "3.10" ]]; then
  RECIPE_PATH="conda/recipes/cudf_kafka"
else
  RECIPE_PATH="conda/recipes/cudf_kafka_py310"
fi

rapids-logger "Building cudf_kafka"

rapids-telemetry-record build-cudf_kafka.log \
    rattler-build build --recipe "${RECIPE_PATH}" \
                    "${RATTLER_ARGS[@]}" \
                    "${RATTLER_CHANNELS[@]}"

rapids-telemetry-record sccache-stats-cudf_kafka.txt sccache --show-adv-stats
sccache --stop-server >/dev/null 2>&1 || true

# remove build_cache directory
rm -rf "$RAPIDS_CONDA_BLD_OUTPUT_DIR"/build_cache

if [[ "${RAPIDS_PY_VERSION}" != "3.10" ]]; then
RAPIDS_PACKAGE_NAME="$(rapids-package-name conda_python cudf --stable --cuda)"
export RAPIDS_PACKAGE_NAME
fi
