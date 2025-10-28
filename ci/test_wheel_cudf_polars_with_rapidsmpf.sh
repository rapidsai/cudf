#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Support invoking test_wheel_cudf_polars_with_rapidsmpf.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../python/cudf_polars

source rapids-configure-sccache
source rapids-date-string

rapids-print-env

rapids-logger "Download cudf_polars and cudf wheels"

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"

# Download cudf_polars, libcudf, and pylibcudf built in previous jobs
CUDF_POLARS_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="cudf_polars_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github python)
LIBCUDF_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="libcudf_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github cpp)
PYLIBCUDF_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="pylibcudf_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github python)

rapids-logger "Install libcudf, pylibcudf, and cudf_polars with experimental extras (includes rapidsmpf from nightly)"

# Install cudf_polars with experimental extras, which will pull in rapidsmpf from nightly
python -m pip install \
    -v \
    "$(echo "${LIBCUDF_WHEELHOUSE}"/libcudf_${RAPIDS_PY_CUDA_SUFFIX}*.whl)" \
    "$(echo "${PYLIBCUDF_WHEELHOUSE}"/pylibcudf_${RAPIDS_PY_CUDA_SUFFIX}*.whl)" \
    "$(echo "${CUDF_POLARS_WHEELHOUSE}"/cudf_polars*.whl)[experimental,test]"

rapids-logger "Check installed versions"
python -c "import cudf_polars; print(f'cudf_polars: {cudf_polars.__version__}')"
python -c "import rapidsmpf; print(f'rapidsmpf: {rapidsmpf.__version__}')"

rapids-logger "Run cudf_polars tests with rapidsmpf"
./ci/run_cudf_polars_with_rapidsmpf_pytests.sh
