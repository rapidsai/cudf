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

PYTEST_IGNORES="--ignore=tests/interchange/test_impl.py \
--ignore=tests/window/test_dtypes.py \
--ignore=tests/strings/test_api.py \
--ignore=tests/window/test_numba.py \
--ignore=tests/window \
--ignore=tests/io/pytables \
--ignore=tests/plotting \
--ignore=tests/scalar \
--ignore=tests/series/test_arithmetic.py \
--ignore=tests/tslibs/test_parsing.py \
--ignore=tests/io/parser/common/test_read_errors.py"

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

# TODO: Remove "not db" once a postgres & mysql container is set up on the CI
PANDAS_CI="1" timeout 30m python -m pytest -p cudf.pandas \
    -v -m "not single_cpu and not db" \
    -k "not test_to_parquet_gcs_new_file and not test_qcut_nat and not test_add and not test_ismethods" \
    --import-mode=importlib \
    ${PYTEST_IGNORES} \
    "$@" || [ $? = 1 ]  # Exit success if exit code was 1 (permit test failures but not other errors)

mv *.json ..
cd ..

rm -rf pandas-testing/pandas-tests/
