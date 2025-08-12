#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Run Pandas unit tests with cudf.pandas.
#
# Usage:
#   run-pandas-tests.sh <pytest args> <path to pandas tests (optional)>
#
# Examples
# Run a single test
#   run-pandas-tests.sh -n auto -v tests/groupby/test_groupby_dropna.py
# Run all tests
#   run-pandas-tests.sh --tb=line --report-log=log.json
#
# This script creates a `pandas-testing` directory if it doesn't exist
#
# If running locally, it's recommended to pass '-m "not slow and not single_cpu and not db"'

set -euo pipefail

EXITCODE=0
trap "EXITCODE=1" ERR
set +e

# Grab the Pandas source corresponding to the version
# of Pandas installed.
PANDAS_VERSION=$(python -c "import pandas; print(pandas.__version__)")

echo "Running Pandas tests for version ${PANDAS_VERSION}"
mkdir -p pandas-testing
cd pandas-testing

if [ ! -d "pandas" ]; then
    git clone https://github.com/pandas-dev/pandas --depth=1 -b "v${PANDAS_VERSION}" pandas
fi

if [ ! -d "pandas-tests" ]; then
    # Copy just the tests out of the Pandas source tree.
    # Not exactly sure why this is needed but Pandas
    # imports fail if we don't do this:
    mkdir -p pandas-tests
    cp -r pandas/pandas/tests pandas-tests/
    # directory layout requirement
    # conftest.py
    # pyproject.toml
    # tests/
    cp pandas/pandas/conftest.py pandas-tests/conftest.py
    # Vendored from pandas/pyproject.toml
    cat > pandas-tests/pyproject.toml << \EOF
[tool.pytest.ini_options]
xfail_strict = true
filterwarnings = [
  # Will be fixed in numba 0.56: https://github.com/numba/numba/issues/7758
  "ignore:`np.MachAr` is deprecated:DeprecationWarning:numba",
]
markers = [
  "single_cpu: tests that should run on a single cpu only",
  "slow: mark a test as slow",
  "network: mark a test as network",
  "db: tests requiring a database (mysql or postgres)",
  "clipboard: mark a pd.read_clipboard test",
  "arm_slow: mark a test as slow for arm64 architecture",
  "skip_ubsan: Tests known to fail UBSAN check",
]
EOF

    # Substitute `pandas.tests` with a relative import.
    # This will depend on the location of the test module relative to
    # the pandas-tests directory.
    for hit in $(find . -iname '*.py' -print0 | xargs -0 grep "pandas.tests" | cut -d ":" -f 1 | sort | uniq); do
        # Get the relative path to the test module
        test_module=$(echo "$hit" | cut -d "/" -f 2-)
        # Get the number of directories to go up
        num_dirs=$(echo "$test_module" | grep -o "/" | wc -l)
        num_dots=$((num_dirs - 2))
        # Construct the relative import
        relative_import=$(printf "%0.s." $(seq 1 $num_dots))
        # Replace the import
        sed -i "s/pandas.tests/${relative_import}/g" "$hit"
    done
fi

# append the contents of patch-confest.py to conftest.py
cat ../python/cudf/cudf/pandas/scripts/conftest-patch.py >> pandas-tests/conftest.py

# Run the tests
cd pandas-tests/


# TODO: Needs motoserver/moto container running on http://localhost:5000
TEST_THAT_NEED_MOTO_SERVER="not test_styler_to_s3 \
and not test_with_s3_url[None] \
and not test_with_s3_url[gzip] \
and not test_with_s3_url[bz2] \
and not test_with_s3_url[zip] \
and not test_with_s3_url[xz] \
and not test_with_s3_url[tar] \
and not test_s3_permission_output[etree] \
and not test_read_s3_jsonl \
and not test_s3_parser_consistency \
and not test_to_s3 \
and not test_parse_public_s3a_bucket \
and not test_parse_public_s3_bucket_nrows \
and not test_parse_public_s3_bucket_chunked \
and not test_parse_public_s3_bucket_chunked_python \
and not test_parse_public_s3_bucket_python \
and not test_infer_s3_compression \
and not test_parse_public_s3_bucket_nrows_python \
and not test_read_s3_fails_private \
and not test_read_csv_handles_boto_s3_object \
and not test_read_csv_chunked_download \
and not test_read_s3_with_hash_in_key \
and not test_read_feather_s3_file_path \
and not test_parse_public_s3_bucket \
and not test_parse_private_s3_bucket \
and not test_parse_public_s3n_bucket \
and not test_read_with_creds_from_pub_bucket \
and not test_read_without_creds_from_pub_bucket \
and not test_from_s3_csv \
and not test_s3_protocols[s3] \
and not test_s3_protocols[s3a] \
and not test_s3_protocols[s3n] \
and not test_s3_parquet \
and not test_s3_roundtrip_explicit_fs \
and not test_s3_roundtrip \
and not test_s3_roundtrip_for_dir[partition_col0] \
and not test_s3_roundtrip_for_dir[partition_col1] \
and not test_s3_roundtrip"

TEST_THAT_CRASH_PYTEST_WORKERS="not test_bitmasks_pyarrow \
and not test_large_string_pyarrow \
and not test_interchange_from_corrected_buffer_dtypes \
and not test_eof_states \
and not test_array_tz \
and not test_resample_empty_dataframe"

# these weakref tests are flaky, since they rely on the timing of the cyclic GC
# We skip them always
TEST_THAT_USE_WEAKREFS="not test_no_reference_cycle"
TEST_THAT_USE_STRING_DTYPE_GROUPBY="not test_string_dtype_all_na"

# TODO: Add reason to skip these tests
TEST_THAT_NEED_REASON_TO_SKIP="not test_groupby_raises_category_on_category \
and not test_constructor_no_pandas_array \
and not test_is_monotonic_na \
and not test_index_contains \
and not test_frame_op_subclass_nonclass_constructor \
and not test_round_trip_current \
and not test_pickle_frame_v124_unpickle_130"

PYTEST_IGNORES=("--ignore=tests/io/parser/common/test_read_errors.py"
                "--ignore=tests/io/test_clipboard.py" # crashes pytest workers (possibly due to fixture patching clipboard functionality)
)


PANDAS_CI="1" python -m pytest -p cudf.pandas \
    --import-mode=importlib \
    -k "$TEST_THAT_NEED_MOTO_SERVER and $TEST_THAT_CRASH_PYTEST_WORKERS and $TEST_THAT_NEED_REASON_TO_SKIP and $TEST_THAT_USE_STRING_DTYPE_GROUPBY and $TEST_THAT_USE_WEAKREFS" \
    "${PYTEST_IGNORES[@]}" \
    "$@"

mv ./*.json ..
cd ..
rm -rf pandas-testing/pandas-tests/
rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
