#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# Run Pandas unit tests with xdf.
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


# Grab the Pandas source corresponding to the version
# of Pandas installed.
PANDAS_VERSION=$(python -c "import pandas; print(pandas.__version__)")

PYTEST_IGNORES="--ignore=tests/io/test_user_agent.py"

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
  # Deprecation gives warning on import during pytest collection
  "ignore:pandas.core.index is deprecated:FutureWarning:importlib",
  "ignore:pandas.util.testing is deprecated:FutureWarning:importlib",
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
PANDAS_CI="1" python -m pytest -p cudf.pandas \
    -m "not single_cpu and not db" \
    -k "not test_overwrite_warns and not test_complex_series_frame_alignment" \
    --durations=50 \
    --import-mode=importlib \
    -o xfail_strict=True \
    ${PYTEST_IGNORES} $@

mv *.json ..
cd ..

rm -rf pandas-testing/pandas-tests/
