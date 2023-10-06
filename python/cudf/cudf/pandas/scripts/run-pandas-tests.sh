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
#   run-pandas-tests.sh --rewrite-imports|--transparent <pytest args> <path to pandas tests (optional)>
#
# Examples
# Run a single test in rewrite-imports mode
#   run-pandas-tests.sh --rewrite-imports -n auto -v tests/groupby/test_groupby_dropna.py
# Run all tests rewriting imports
#   run-pandas-tests.sh --rewrite-imports --tb=line --report-log=log.json
# Run all tests in transparent mode
#   run-pandas-tests.sh --transparent --tb=line --report-log=log.json
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


XDF_MODE=${1}
# Consume first argument
shift

if [ -f "pandas-tests/.xdf-run-mode" ]; then
    LAST_RUN_MODE=$(<pandas-tests/.xdf-run-mode)
    if [[ "${LAST_RUN_MODE}" != "${XDF_MODE}" ]]; then
        rm -rf pandas-tests
    fi
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
    cat ../cudf/pandas/scripts/conftest-patch.py >> pandas-tests/conftest.py

    # Substitute `pandas.tests` with a relative import.
    # This will depend on the location of the test module relative to
    # the pandas-tests directory.
    for hit in $(find . -iname *.py | xargs grep "pandas.tests" | cut -d ":" -f 1 | sort | uniq); do
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


if [[ "${XDF_MODE}" == "--rewrite-imports" ]]; then
    # Substitute `xdf` for `pandas` in the tests
    find pandas-tests/ -iname "*.py" | xargs sed -i 's/import\ pandas/import\ cudf.pandas.pandas/g'
    find pandas-tests/ -iname "*.py" | xargs sed -i 's/from\ pandas/from\ cudf.pandas.pandas/g'
    find pandas-tests/ -iname "*.py" | xargs sed -i 's/cudf.pandas.pandas_dtype/pandas_dtype/g'

    EXTRA_PYTEST_ARGS=""
elif [[ "${XDF_MODE}" == "--transparent" ]]; then
    EXTRA_PYTEST_ARGS="-p cudf.pandas"
else
    echo "Unknown XDF mode ${XDF_MODE}, expecting '--rewrite-imports' or '--transparent'"
    exit 1
fi

# append the contents of patch-confest.py to conftest.py
cat ../cudf/pandas/scripts/conftest-patch.py >> pandas-tests/conftest.py

echo -n "${XDF_MODE}" > pandas-tests/.xdf-run-mode

# Run the tests
cd pandas-tests/

# TODO: Get a postgres & mysql container set up on the CI
PANDAS_CI="1" python -m pytest -m "not single_cpu and not db" --durations=50 --import-mode=importlib -o xfail_strict=True ${PYTEST_IGNORES} ${EXTRA_PYTEST_ARGS} $@

mv *.json ..
cd ..

rm -rf pandas-testing/pandas-tests/
