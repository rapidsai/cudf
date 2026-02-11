# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

"""Module used for global configuration of benchmarks.

This file contains global definitions that are important for configuring all
benchmarks such as fixture sizes. In addition, this file supports the following
features:
    - Defining the CUDF_BENCHMARKS_USE_PANDAS environment variable will change
      all benchmarks to run with pandas instead of cudf (and numpy instead of
      cupy). This feature enables easy comparisons of benchmarks between cudf
      and pandas. All common modules (cudf, cupy) should be imported from here
      by benchmark modules to allow configuration if needed.
    - Defining CUDF_BENCHMARKS_DEBUG_ONLY will set global configuration
      variables to avoid running large benchmarks, instead using minimal values
      to simply ensure that benchmarks are functional.

This file is also where standard pytest hooks should be overridden. While these
definitions typically belong in conftest.py, since any of the above environment
variables could affect test collection or other properties, we must define them
in this file and import them in conftest.py to ensure that they are handled
appropriately.
"""

import os
import sys

# Environment variable-based configuration of benchmarking pandas or cudf.
collect_ignore = []
if "CUDF_BENCHMARKS_USE_PANDAS" in os.environ:
    import numpy as cupy
    import pandas as cudf

    # cudf internals offer no pandas compatibility guarantees, and we also
    # never need to compare those benchmarks to pandas.
    collect_ignore.append("internal/")

    # Also filter out benchmarks of APIs that are not compatible with pandas.
    def is_pandas_compatible(item):
        return all(m.name != "pandas_incompatible" for m in item.own_markers)

    def pytest_collection_modifyitems(session, config, items):
        items[:] = list(filter(is_pandas_compatible, items))

else:
    import cupy  # noqa: F401

    import cudf  # noqa: F401

    def pytest_collection_modifyitems(session, config, items):
        pass


def pytest_sessionstart(session):
    """Add the common files to the path for all tests to import."""
    sys.path.insert(0, os.path.join(os.getcwd(), "common"))


def pytest_sessionfinish(session, exitstatus):
    """Clean up sys.path after exit."""
    if "common" in sys.path[0]:
        del sys.path[0]


# Constants used to define benchmarking standards. We always generate all the fixtures,
# but benchmarks can choose which ones to use.
NUM_ROWS_FIXTURES = [100, 10_000, 1_000_000]
NUM_COLS_FIXTURES = [1, 6, 20]

# When in debug mode, limit the number of rows and columns to the lowest values. It must
# be a subset of the values defined above to ensure that debug runs are actually
# validating that non-debug runs will work.
NUM_ROWS = NUM_ROWS_FIXTURES
NUM_COLS = NUM_COLS_FIXTURES
if "CUDF_BENCHMARKS_DEBUG_ONLY" in os.environ:
    NUM_ROWS = NUM_ROWS[:1]
    NUM_COLS = NUM_COLS[:1]
