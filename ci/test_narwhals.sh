#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION.

# Support invoking test_python_cudf.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../ || exit 1

# Common setup steps shared by Python test jobs
source ./ci/test_python_common.sh test_python_narwhals

rapids-logger "Check GPU usage"
nvidia-smi
rapids-print-env
EXITCODE=0
trap "EXITCODE=1" ERR
set +e

rapids-logger "pytest narwhals"
git clone https://github.com/narwhals-dev/narwhals --depth=1 -b stable
pushd narwhals || exit 1
rapids-pip-retry install -U -e ".[dev]"

rapids-logger "Check narwhals versions"
python -c "import narwhals; print(narwhals.show_versions())"

rapids-logger "Run narwhals tests for cuDF"
python -m pytest \
    --cache-clear \
    --junitxml="${RAPIDS_TESTS_DIR}/junit-cudf-narwhals.xml" \
    -p cudf.testing.narwhals_test_plugin \
    --numprocesses=8 \
    --dist=worksteal \
    --constructors=cudf

rapids-logger "Run narwhals tests for cuDF Polars"
NARWHALS_POLARS_GPU=1 python -m pytest \
    --cache-clear \
    --junitxml="${RAPIDS_TESTS_DIR}/junit-cudf-polars-narwhals.xml" \
    --numprocesses=8 \
    --dist=worksteal \
    --constructors=polars[lazy]

rapids-logger "Run narwhals tests for cuDF Pandas"
# TODO: Investigate which of the tests we can avoid skipping
# Tracking Issue: https://github.com/rapidsai/cudf/issues/18248
NARWHALS_DEFAULT_CONSTRUCTORS=pandas python -m pytest \
    -p cudf.pandas \
    --cache-clear \
    --junitxml="${RAPIDS_TESTS_DIR}/junit-cudf-pandas-narwhals.xml" \
    -k "not ( \
        test_pandas_object_series or \
        test_is_finite_expr or \
        test_is_finite_series or \
        test_array_dunder_with_copy or \
        test_maybe_convert_dtypes_pandas or \
        test_to_arrow or \
        test_to_arrow_with_nulls or \
        test_sumh_transformations or \
        test_dask_order_dependent_ops or \
        test_q1 \
    )" \
    --numprocesses=8 \
    --dist=worksteal

popd || exit 1

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
