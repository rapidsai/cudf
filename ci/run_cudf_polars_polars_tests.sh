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

DESELECTED_TESTS=$(printf -- " --deselect %s" "${DESELECTED_TESTS[@]}")
python -m pytest \
       --import-mode=importlib \
       --cache-clear \
       -m "" \
       -p cudf_polars.testing.plugin \
       -v \
       --tb=short \
       ${DESELECTED_TESTS} \
       "$@" \
       py-polars/tests
