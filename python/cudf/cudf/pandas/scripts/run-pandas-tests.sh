#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES.
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

set -euo pipefail

# Grab the Pandas source corresponding to the version
# of Pandas installed.
PANDAS_VERSION=$(python -c "import pandas; print(pandas.__version__)")

PYTEST_IGNORES="--ignore=tests/io/test_user_agent.py --ignore=tests/interchange/test_impl.py --ignore=tests/window/test_dtypes.py --ignore=tests/strings/test_api.py --ignore=tests/window/test_numba.py"

mkdir -p pandas-testing
cd pandas-testing

if [ ! -d "pandas" ]; then
    git clone https://github.com/pandas-dev/pandas
fi
cd pandas && git clean -fdx && git checkout v$PANDAS_VERSION && cd ../


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
  "error:Sparse:FutureWarning",
  "error:The SparseArray:FutureWarning",
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
  "arraymanager: mark a test to run with ArrayManager enabled",
]
EOF
    # append the contents of patch-confest.py to conftest.py
    cat ../python/cudf/cudf/pandas/scripts/conftest-patch.py >> pandas-tests/conftest.py

    # Substitute `pandas.tests` with a relative import.
    # This will depend on the location of the test module relative to
    # the pandas-tests directory.
    for hit in $(find . -iname '*.py' | xargs grep "pandas.tests" | cut -d ":" -f 1 | sort | uniq); do
        # Get the relative path to the test module
        test_module=$(echo $hit | cut -d "/" -f 2-)
        # Get the number of directories to go up
        num_dirs=$(echo $test_module | grep -o "/" | wc -l)
        num_dots=$(($num_dirs - 2))
        # Construct the relative import
        relative_import=$(printf "%0.s." $(seq 1 $num_dots))
        # Replace the import
        sed -i "s/pandas.tests/${relative_import}/g" $hit
    done
fi

# append the contents of patch-confest.py to conftest.py
cat ../python/cudf/cudf/pandas/scripts/conftest-patch.py >> pandas-tests/conftest.py

# Run the tests
cd pandas-tests/

# TODO: Get a postgres & mysql container set up on the CI
# test_overwrite_warns unsafely patchs over Series.mean affecting other tests when run in parallel
# test_complex_series_frame_alignment randomly selects a DataFrames and axis to test but particular random selection(s) always fails
# test_numpy_ufuncs_basic compares floating point values to unbounded precision, sometimes leading to failures
TEST_NUMPY_UFUNCS_BASIC_FLAKY="not test_numpy_ufuncs_basic[float-exp] \
and not test_numpy_ufuncs_basic[float-exp2] \
and not test_numpy_ufuncs_basic[float-expm1] \
and not test_numpy_ufuncs_basic[float-log] \
and not test_numpy_ufuncs_basic[float-log2] \
and not test_numpy_ufuncs_basic[float-log10] \
and not test_numpy_ufuncs_basic[float-log1p] \
and not test_numpy_ufuncs_basic[float-sqrt] \
and not test_numpy_ufuncs_basic[float-sin] \
and not test_numpy_ufuncs_basic[float-cos] \
and not test_numpy_ufuncs_basic[float-tan] \
and not test_numpy_ufuncs_basic[float-arcsin] \
and not test_numpy_ufuncs_basic[float-arccos] \
and not test_numpy_ufuncs_basic[float-arctan] \
and not test_numpy_ufuncs_basic[float-sinh] \
and not test_numpy_ufuncs_basic[float-cosh] \
and not test_numpy_ufuncs_basic[float-tanh] \
and not test_numpy_ufuncs_basic[float-arcsinh] \
and not test_numpy_ufuncs_basic[float-arccosh] \
and not test_numpy_ufuncs_basic[float-arctanh] \
and not test_numpy_ufuncs_basic[float-deg2rad] \
and not test_numpy_ufuncs_basic[float-rad2deg] \
and not test_numpy_ufuncs_basic[num_float64-exp] \
and not test_numpy_ufuncs_basic[num_float64-exp2] \
and not test_numpy_ufuncs_basic[num_float64-expm1] \
and not test_numpy_ufuncs_basic[num_float64-log] \
and not test_numpy_ufuncs_basic[num_float64-log2] \
and not test_numpy_ufuncs_basic[num_float64-log10] \
and not test_numpy_ufuncs_basic[num_float64-log1p] \
and not test_numpy_ufuncs_basic[num_float64-sqrt] \
and not test_numpy_ufuncs_basic[num_float64-sin] \
and not test_numpy_ufuncs_basic[num_float64-cos] \
and not test_numpy_ufuncs_basic[num_float64-tan] \
and not test_numpy_ufuncs_basic[num_float64-arcsin] \
and not test_numpy_ufuncs_basic[num_float64-arccos] \
and not test_numpy_ufuncs_basic[num_float64-arctan] \
and not test_numpy_ufuncs_basic[num_float64-sinh] \
and not test_numpy_ufuncs_basic[num_float64-cosh] \
and not test_numpy_ufuncs_basic[num_float64-tanh] \
and not test_numpy_ufuncs_basic[num_float64-arcsinh] \
and not test_numpy_ufuncs_basic[num_float64-arccosh] \
and not test_numpy_ufuncs_basic[num_float64-arctanh] \
and not test_numpy_ufuncs_basic[num_float64-deg2rad] \
and not test_numpy_ufuncs_basic[num_float64-rad2deg] \
and not test_numpy_ufuncs_basic[num_float32-exp] \
and not test_numpy_ufuncs_basic[num_float32-exp2] \
and not test_numpy_ufuncs_basic[num_float32-expm1] \
and not test_numpy_ufuncs_basic[num_float32-log] \
and not test_numpy_ufuncs_basic[num_float32-log2] \
and not test_numpy_ufuncs_basic[num_float32-log10] \
and not test_numpy_ufuncs_basic[num_float32-log1p] \
and not test_numpy_ufuncs_basic[num_float32-sqrt] \
and not test_numpy_ufuncs_basic[num_float32-sin] \
and not test_numpy_ufuncs_basic[num_float32-cos] \
and not test_numpy_ufuncs_basic[num_float32-tan] \
and not test_numpy_ufuncs_basic[num_float32-arcsin] \
and not test_numpy_ufuncs_basic[num_float32-arccos] \
and not test_numpy_ufuncs_basic[num_float32-arctan] \
and not test_numpy_ufuncs_basic[num_float32-sinh] \
and not test_numpy_ufuncs_basic[num_float32-cosh] \
and not test_numpy_ufuncs_basic[num_float32-tanh] \
and not test_numpy_ufuncs_basic[num_float32-arcsinh] \
and not test_numpy_ufuncs_basic[num_float32-arccosh] \
and not test_numpy_ufuncs_basic[num_float32-arctanh] \
and not test_numpy_ufuncs_basic[num_float32-deg2rad] \
and not test_numpy_ufuncs_basic[num_float32-rad2deg] \
and not test_numpy_ufuncs_basic[nullable_float-exp] \
and not test_numpy_ufuncs_basic[nullable_float-exp2] \
and not test_numpy_ufuncs_basic[nullable_float-expm1] \
and not test_numpy_ufuncs_basic[nullable_float-log] \
and not test_numpy_ufuncs_basic[nullable_float-log2] \
and not test_numpy_ufuncs_basic[nullable_float-log10] \
and not test_numpy_ufuncs_basic[nullable_float-log1p] \
and not test_numpy_ufuncs_basic[nullable_float-sqrt] \
and not test_numpy_ufuncs_basic[nullable_float-sin] \
and not test_numpy_ufuncs_basic[nullable_float-cos] \
and not test_numpy_ufuncs_basic[nullable_float-tan] \
and not test_numpy_ufuncs_basic[nullable_float-arcsin] \
and not test_numpy_ufuncs_basic[nullable_float-arccos] \
and not test_numpy_ufuncs_basic[nullable_float-arctan] \
and not test_numpy_ufuncs_basic[nullable_float-sinh] \
and not test_numpy_ufuncs_basic[nullable_float-cosh] \
and not test_numpy_ufuncs_basic[nullable_float-tanh] \
and not test_numpy_ufuncs_basic[nullable_float-arcsinh] \
and not test_numpy_ufuncs_basic[nullable_float-arccosh] \
and not test_numpy_ufuncs_basic[nullable_float-arctanh] \
and not test_numpy_ufuncs_basic[nullable_float-deg2rad] \
and not test_numpy_ufuncs_basic[nullable_float-rad2deg]"

PANDAS_CI="1" python -m pytest -p cudf.pandas \
    -v -m "not single_cpu and not db" \
    -k "not test_overwrite_warns and not test_complex_series_frame_alignment and not test_to_parquet_gcs_new_file and not test_qcut_nat and not test_add and not test_ismethods and $TEST_NUMPY_UFUNCS_BASIC_FLAKY" \
    --durations=50 \
    --import-mode=importlib \
    -o xfail_strict=True \
    ${PYTEST_IGNORES} \
    "$@" || [ $? = 1 ]  # Exit success if exit code was 1 (permit test failures but not other errors)

mv *.json ..
cd ..

rm -rf pandas-testing/pandas-tests/
