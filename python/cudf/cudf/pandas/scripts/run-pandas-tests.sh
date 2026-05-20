#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Run Pandas unit tests with cudf.pandas.
#
# Usage:
#   run-pandas-tests.sh <pytest args>
#
# Examples
# Run a single test
#   run-pandas-tests.sh -n auto -v pandas/pandas_src/tests/groupby/test_groupby_dropna.py
# Run all tests
#   run-pandas-tests.sh --tb=line --report-log=log.json
#
# This script creates a `pandas-testing` directory if it doesn't exist and
# clones the pandas repository inside it.  The only modification made to the
# clone is renaming its inner package directory from `pandas/` to
# `pandas_src/`: this prevents pytest from loading `pandas/conftest.py` as
# the module `pandas.conftest`, which would otherwise collide with the
# installed (cudf.pandas-wrapped) `pandas` module in `sys.modules` and cause
# a recursion in cudf.pandas's fast-slow proxy during test collection.
#
# The pandas test suite is then invoked directly against
# `pandas/pandas_src/tests/` with two pytest plugins loaded:
#
#   -p cudf.pandas
#   -p cudf.pandas._pandas_tests_plugin
#
# If running locally, it's recommended to pass '-m "not slow and not single_cpu and not db"'

set -euo pipefail

EXITCODE=0
trap "EXITCODE=1" ERR
set +e

# Grab the Pandas source corresponding to the version of Pandas installed.
PANDAS_VERSION=$(python -c "import pandas; print(pandas.__version__)")

echo "Running Pandas tests for version ${PANDAS_VERSION}"
mkdir -p pandas-testing
cd pandas-testing

if [ ! -d "pandas" ]; then
    git clone https://github.com/pandas-dev/pandas --depth=1 \
        -b "v${PANDAS_VERSION}" pandas
    # Rename the inner package directory to avoid `pandas.conftest` clashing
    # with the installed (cudf.pandas-wrapped) `pandas` module.
    mv pandas/pandas pandas/pandas_src
fi

# Pytest auto-discovers pandas/pyproject.toml so rootdir is set to
# `pandas-testing/pandas/` and node ids come out as
# `pandas_src/tests/<...>::<test>`.  The only ini option we override is
# `filterwarnings`: pandas's default of `["error:::pandas", ...]` converts
# pandas-origin warnings into test failures, which is incompatible with the
# additional warnings that cudf.pandas raises.
PANDAS_CI="1" python -m pytest \
    --override-ini="filterwarnings=" \
    --import-mode=importlib \
    -p cudf.pandas \
    -p cudf.pandas._pandas_tests_plugin \
    pandas/pandas_src/tests/ \
    "$@"

echo "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
