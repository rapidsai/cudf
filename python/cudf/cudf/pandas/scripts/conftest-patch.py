# SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib
import json
import os
import sys
import traceback
from collections import defaultdict
from functools import wraps

import pytest


def replace_kwargs(new_kwargs):
    def wrapper(func):
        @wraps(func)
        def wrapped(*args, **kwargs):
            kwargs.update(new_kwargs)
            return func(*args, **kwargs)

        return wrapped

    return wrapper


@contextlib.contextmanager
def null_assert_warnings(*args, **kwargs):
    try:
        yield []
    finally:
        pass


@pytest.fixture(scope="session", autouse=True)  # type: ignore
def patch_testing_functions():
    tm.assert_produces_warning = null_assert_warnings  # noqa: F821
    pytest.raises = replace_kwargs({"match": None})(pytest.raises)


# Dictionary to store function call counts
function_call_counts = {}  # type: ignore

# The specific functions to track
FUNCTION_NAME = {"_slow_function_call", "_fast_function_call"}


def find_pytest_file(frame):
    stack = traceback.extract_stack()
    absolute_paths = [frame.filename for frame in stack]
    for file in absolute_paths:
        if "pandas-testing/pandas-tests/tests" in file and file.rsplit("/", 1)[
            -1
        ].startswith("test_"):
            return str(file).rsplit("pandas-tests/", 1)[-1]
    return None


def trace_calls(frame, event, arg):
    if event != "call":
        return
    code = frame.f_code
    func_name = code.co_name

    if func_name in FUNCTION_NAME:
        filename = find_pytest_file(frame)
        if filename is None:
            return
        if filename not in function_call_counts:
            function_call_counts[filename] = defaultdict(int)
        function_call_counts[filename][func_name] += 1


def pytest_sessionstart(session):
    # Set the profile function to trace calls
    sys.setprofile(trace_calls)


def pytest_sessionfinish(session, exitstatus):
    # Remove the profile function
    sys.setprofile(None)


@pytest.hookimpl(trylast=True)
def pytest_unconfigure(config):
    if hasattr(config, "workerinput"):
        # Running in xdist worker, write the counts before exiting
        worker_id = config.workerinput["workerid"]
        output_file = f"function_call_counts_worker_{worker_id}.json"
        with open(output_file, "w") as f:
            json.dump(function_call_counts, f, indent=4)
        print(f"Function call counts have been written to {output_file}")


sys.path.append(os.path.dirname(__file__))
