# SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib
import json
import multiprocessing
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
    tm.assert_produces_warning = null_assert_warnings
    pytest.raises = replace_kwargs({"match": None})(pytest.raises)


# Dictionary to store function call counts
manager = multiprocessing.Manager()
function_call_counts = defaultdict(int)  # type: ignore

# The specific function to track
FUNCTION_NAME = {"_slow_function_call", "_fast_function_call"}


def find_pytest_file(frame):
    stack = traceback.extract_stack()
    absolute_paths = [frame.filename for frame in stack]
    for file in absolute_paths:
        if "pandas-testing/pandas-tests/tests" in file and file.rsplit("/", 1)[
            -1
        ].startswith("test_"):
            return file
    return None
    # new_f = frame
    # while new_f:
    #     if "pandas-testing/pandas-tests/tests" in new_f.f_globals.get("__file__", ""):
    #         return os.path.abspath(new_f.f_globals.get("__file__", ""))
    #     new_f = new_f.f_back
    # return None


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


# @pytest.hookimpl(tryfirst=True)
# def pytest_runtest_setup(item):
#     # Check if this is the first test in the file
#     if item.nodeid.split("::")[0] != getattr(
#         pytest_runtest_setup, "current_file", None
#     ):
#         # If it's a new file, reset the function call counts
#         global function_call_counts
#         function_call_counts = defaultdict(int)
#         pytest_runtest_setup.current_file = item.nodeid.split("::")[0]


# @pytest.hookimpl(trylast=True)
# def pytest_runtest_teardown(item, nextitem):
#     # Check if this is the last test in the file
#     if (
#         nextitem is None
#         or nextitem.nodeid.split("::")[0] != item.nodeid.split("::")[0]
#     ):
#         # Write the function call counts to a file
#         worker_id = os.getenv("PYTEST_XDIST_WORKER", "master")
#         output_file = f'{item.nodeid.split("::")[0].replace("/", "__")}_{worker_id}_metrics.json'
#         # if os.path.exists(output_file):
#         #     output_file = f'{item.nodeid.split("::")[0].replace("/", "__")}_{worker_id}_metrics_1.json'
#         with open(output_file, "w") as f:
#             json.dump(dict(function_call_counts), f, indent=4)
#         print(f"Function call counts have been written to {output_file}")


# @pytest.hookimpl(tryfirst=True)
# def pytest_configure(config):
#     if hasattr(config, "workerinput"):
#         # Running in xdist worker
#         global function_call_counts
#         function_call_counts = defaultdict(int)


@pytest.hookimpl(trylast=True)
def pytest_unconfigure(config):
    if hasattr(config, "workerinput"):
        # Running in xdist worker
        worker_id = config.workerinput["workerid"]
        output_file = f"function_call_counts_worker_{worker_id}.json"
        with open(output_file, "w") as f:
            json.dump(dict(function_call_counts), f, indent=4)
        # print(f"Function call counts have been written to {output_file}")


sys.path.append(os.path.dirname(__file__))
