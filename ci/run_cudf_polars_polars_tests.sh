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
    "tests/unit/test_polars_import.py::test_fork_safety" # test started to hang in polars-1.14
    "tests/unit/operations/test_join.py::test_join_4_columns_with_validity" # fails in some systems, see https://github.com/pola-rs/polars/issues/19870
    "tests/unit/io/test_csv.py::test_read_web_file" # fails in rockylinux8 due to SSL CA issues
)

if [[ $(arch) == "aarch64" ]]; then
    # The binary used for TPC-H generation is compiled for x86_64, not aarch64.
    DESELECTED_TESTS+=("tests/benchmark/test_pdsh.py::test_pdsh")
    # The connectorx package is not available on arm
    DESELECTED_TESTS+=("tests/unit/io/database/test_read.py::test_read_database")
    # The necessary timezone information cannot be found in our CI image.
    DESELECTED_TESTS+=("tests/unit/io/test_parquet.py::test_parametric_small_page_mask_filtering")
    DESELECTED_TESTS+=("tests/unit/testing/test_assert_series_equal.py::test_assert_series_equal_parametric")
    DESELECTED_TESTS+=("tests/unit/operations/test_join.py::test_join_4_columns_with_validity")
else
    # Ensure that we don't run dbgen when it uses newer symbols than supported by the glibc version in the CI image.
    # Allow errors since any of these commands could produce empty results that would cause the script to fail.
    set +e
    glibc_minor_version=$(ldd --version | head -1 | grep -o "[0-9]\.[0-9]\+" | tail -1 | cut -d '.' -f2)
    latest_glibc_symbol_found=$(nm py-polars/tests/benchmark/data/pdsh/dbgen/dbgen | grep GLIBC | grep -o "[0-9]\.[0-9]\+" | sort --version-sort | tail -1 | cut -d "." -f 2)
    set -e
    if [[ ${glibc_minor_version} -lt ${latest_glibc_symbol_found} ]]; then
        DESELECTED_TESTS+=("tests/benchmark/test_pdsh.py::test_pdsh")
    fi
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
