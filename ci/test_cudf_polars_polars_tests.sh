#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

source rapids-init-pip

rapids-logger "Download wheels"

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"
CUDF_POLARS_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="cudf_polars_${RAPIDS_PY_CUDA_SUFFIX}" RAPIDS_PY_WHEEL_PURE="1" rapids-download-wheels-from-github python)
LIBCUDF_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="libcudf_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github cpp)
PYLIBCUDF_WHEELHOUSE=$(rapids-download-from-github "$(rapids-package-name "wheel_python" pylibcudf --stable --cuda "$RAPIDS_CUDA_VERSION")")

rapids-logger "Install libcudf, pylibcudf and cudf_polars"
rapids-pip-retry install \
    -v \
    "$(echo "${CUDF_POLARS_WHEELHOUSE}"/cudf_polars_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)[test]" \
    "$(echo "${LIBCUDF_WHEELHOUSE}"/libcudf_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)" \
    "$(echo "${PYLIBCUDF_WHEELHOUSE}"/pylibcudf_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)"


TAG=$(python -c 'import polars; print(f"py-{polars.__version__}")')
rapids-logger "Clone polars to ${TAG}"
git clone https://github.com/pola-rs/polars.git --branch "${TAG}" --depth 1

# Install requirements for running polars tests
rapids-logger "Install polars test requirements"
# We don't need to pick up dependencies from polars-cloud, so we remove it.
sed -i '/^polars-cloud$/d' polars/py-polars/requirements-dev.txt
# Deltalake release 1.2.0 contains breaking changes for Polars.
# Tracking issue: https://github.com/pola-rs/polars/issues/24872
sed -i 's/^deltalake>=1.1.4/deltalake>=1.1.4,<1.2.0/' polars/py-polars/requirements-dev.txt
# pyiceberg depends on a non-documented attribute of pydantic.
# AttributeError: 'pydantic_core._pydantic_core.ValidationInfo' object has no attribute 'current_schema_id'
sed -i 's/^pydantic>=2.0.0.*/pydantic>=2.0.0,<2.12.0/' polars/py-polars/requirements-dev.txt
# Iceberg tests include a call to a deprecated in 0.10.0
# See https://github.com/pola-rs/polars/pull/25854
# Ignore the warning for now, but update the minimum
# iceberg pinning after the 0.11.0 release.
sed -i 's/warnings.simplefilter.*PydanticDeprecatedSince212/# &/' polars/py-polars/tests/unit/io/test_iceberg.py
sed -i '/PydanticDeprecatedSince212/a \    warnings.simplefilter("ignore", DeprecationWarning)' polars/py-polars/tests/unit/io/test_iceberg.py

# https://github.com/pola-rs/polars/issues/25772
# Remove upper bound on aiosqlite once we support polars >1.36.1
sed -i 's/^aiosqlite/aiosqlite>=0.21.0,<0.22.0/' polars/py-polars/requirements-dev.txt

# Remove upper bound on pandas once we support 3.0.0+
sed -i 's/^pandas$/pandas>=2.0,<2.4.0/' polars/py-polars/requirements-dev.txt

# Remove upper bound on pandas-stubs once we support 3.0.0+
sed -i 's/^pandas-stubs/pandas-stubs<3/' polars/py-polars/requirements-dev.txt

# Pyparsing release 3.3.0 deprecates the enablePackrat method, which is used by the
# version of pyiceberg that polars is currently pinned to. We can remove this skip
# when we move to a newer version of polars using a pyiceberg where this issue is fixed
# Currently pyparsing is only a transitive dependency via pyiceberg, so we just
# tack on the constrained dependency at the end of the file since there is no
# existing dependency to rewrite.
echo "pyparsing>=3.0.0,<3.3.0" >> polars/py-polars/requirements-dev.txt

rapids-pip-retry install -r polars/py-polars/requirements-dev.txt -r polars/py-polars/requirements-ci.txt

# shellcheck disable=SC2317
function set_exitcode()
{
    EXITCODE=$?
}
EXITCODE=0
trap set_exitcode ERR
set +e

rapids-logger "Run polars tests"
timeout 30m ./ci/run_cudf_polars_polars_tests.sh

trap ERR
set -e

if [ ${EXITCODE} != 0 ]; then
    rapids-logger "Running polars test suite FAILED: exitcode ${EXITCODE}"
else
    rapids-logger "Running polars test suite PASSED"
fi
exit ${EXITCODE}
