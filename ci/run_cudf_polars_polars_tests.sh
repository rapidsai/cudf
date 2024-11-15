#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

set -euo pipefail

# Support invoking run_cudf_polars_pytests.sh outside the script directory
# Assumption, polars has been cloned in the root of the repo.
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../polars/

DESELECTED_TESTS=(
    "tests/unit/test_polars_import.py::test_polars_import" # relies on a polars built in place
    "tests/unit/streaming/test_streaming_sort.py::test_streaming_sort[True]" # relies on polars built in debug mode
    "tests/unit/test_cpu_check.py::test_check_cpu_flags_skipped_no_flags" # Mock library error
    "tests/docs/test_user_guide.py" # No dot binary in CI image
)

if [[ $(arch) == "aarch64" ]]; then
    # The binary used for TPC-H generation is compiled for x86_64, not aarch64.
    DESELECTED_TESTS+=("tests/benchmark/test_pdsh.py::test_pdsh")
    # The connectorx package is not available on arm
    DESELECTED_TESTS+=("ests/unit/io/database/test_read.py::test_read_database")
    # The necessary timezone information cannot be found in our CI image.
    DESELECTED_TESTS+=("ests/unit/io/test_parquet.py::test_parametric_small_page_mask_filtering")
    DESELECTED_TESTS+=("ests/unit/testing/test_assert_series_equal.py::test_assert_series_equal_parametric")
    DESELECTED_TESTS+=("y-polars/tests/unit/operations/test_join.py::test_join_4_columns_with_validity")
fi

DESELECTED_TESTS=$(printf -- " --deselect %s" "${DESELECTED_TESTS[@]}")
python -m pytest \
       --import-mode=importlib \
       --cache-clear \
       -m "" \
       -p cudf_polars.testing.plugin \
       -v \
       --tb=native \
       ${DESELECTED_TESTS} \
       "$@" \
       py-polars/tests
