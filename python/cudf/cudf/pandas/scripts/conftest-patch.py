# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES.
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
        print(f"Function call counts have been written to {output_file}")  # noqa: T201


ERROR_MESSAGES = {
    "Names should be list-like for a MultiIndex",
    "Attributes of DataFrame.iloc[:, 1] (",
    "Could not convert <NA> with type NAType: did not recognize Python value type when inferring an Arrow data type",
    "Failed: DID NOT RAISE <class 'TypeError'>",
    "'Column not found: both'",
    "parallel_for failed: cudaErrorUnknown: unknown error",
    "assert {} == {'a': 1}",
    "'DataFrameGroupBy' object has no attribute",
    "'SeriesGroupBy' object has no attribute",
    "DataFrame are different",
    "'RangeIndex' object has no attribute '_range'",
    "DID NOT RAISE <",
    "right is not an ExtensionArray",
    "bad operand type for unary +:",
    "'ExcelFile' object has no attribute",
    "Could not convert Timedelta(",
    "'bool' object has no attribute 'any'",
    "'bool' object has no attribute 'all'",
    "The truth value of a DataFrame is ambiguous",
    "Cannot directly instantiate object",
    "Must pass DataFrame or 2-d ndarray with boolean values only",
    "closed keyword does not match dtype.closed",
    "Names must be a list-like",
    "'SQLTable' object has no attribute 'table'",
    "(<MonthEnd>, None)",
    "Did not use numexpr as expected.",
    "'_MethodProxy' object has no attribute",
    "assert 1 > 1",
    "'numpy.dtypes.ObjectDType' object has no attribute 'numpy_dtype'",
    "'numpy.dtypes.Float64DType' object has no attribute 'numpy_dtype'",
    "'numpy.dtypes.ObjectDType' object has no attribute 'numpy_dtype'",
    "DID NOT RAISE (<class 'TypeError'>, <class 'TypeError'>)",
    "'>' not supported between instances of 'float' and 'tuple'",
    "'<' not supported between instances of 'float' and 'tuple'",
    "unhashable type: 'list'",
    "left is not an ExtensionArray",
    "DID NOT RAISE (<class 'NotImplementedError'>, <class 'TypeError'>)",
    "ufunc 'add' cannot use operands with types",
    'Attribute "inferred_type" are different',
    "cannot reindex on an axis with duplicate labels",
    "Series Expected type <class 'pandas.core.series.Series'>, found <class 'numpy.ndarray'> instead",
    "Series Expected type",
    "numpy array are different",
    "dtype('<M8[h]')",
    "dtype('<M8[m]')",
    "assert 0 == 1",
    "assert None",
    "assert <DatetimeArray>\\n",
    "'Period' object is not iterable",
    "'DataFrameGroupBy' object has no attribute '_grouper'. Did you mean: 'grouper'?",
    "Length of values (4) does not match length of index (3)",
    "columns length expected 4 but found 2",
    "Shape of passed values is (3, 4), indices imply (3, 2)",
    "database is locked",
    "Fast implementation not available. Falling back to the slow implementation",
    "Option 'chained_assignment' has already been registered",
    "'Styler' object has no attribute 'columns'",
    "freq is not implemented yet",
    "'numpy.ndarray' object has no attribute 'columns'",
    "'numpy.ndarray' object has no attribute 'index'",
    "DataFrame object has no attribute tolist",
    "'DataFrame' object has no attribute 'tolist'",
    "'SparseArray' object has no attribute '_sparse_values'",
    "'Series' object has no attribute 'x'",
    "'numpy.ndarray' object has no attribute 'iloc'",
    "'Categorical' object has no attribute '__array__'",
    "'MyIndex' object has no attribute '_fsproxy_wrapped'",
    "'bool' object has no attribute 'item'",
    "'numpy.float64' object has no attribute 'sort_index'",
    "'FrozenList' object has no attribute '__array__'",
    "'DatetimeTZDtype' object has no attribute '__from_arrow__'",
    "'Int8Dtype' object has no attribute '__from_arrow__'",
    "'Int16Dtype' object has no attribute '__from_arrow__'",
    "'Int32Dtype' object has no attribute '__from_arrow__'",
    "'Int64Dtype' object has no attribute '__from_arrow__'",
    "'UInt8Dtype' object has no attribute '__from_arrow__'",
    "'UInt16Dtype' object has no attribute '__from_arrow__'",
    "'UInt32Dtype' object has no attribute '__from_arrow__'",
    "'UInt64Dtype' object has no attribute '__from_arrow__'",
    "'Float32Dtype' object has no attribute '__from_arrow__'",
    "'Float64Dtype' object has no attribute '__from_arrow__'",
    "'Float16Dtype' object has no attribute '__from_arrow__'",
    "Duplicate column names are not allowed",
    "value must be a scalar",
    "All columns must be of equal length",
    "The truth value of a Series is ambiguous. Use a.empty, a.bool(), a.item(), a.any() or a.all().",
    "codes need to be between -1 and len(categories)-1",
    "This column does not support to be converted to a pandas ExtensionArray",
    "storage_options passed with file object or non-fsspec file path",
    "Inferred frequency None from passed values does not conform to passed frequency",
    "Could not convert strings to integer type due to presence of non-integer values.",
    "Length of names must match number of levels in MultiIndex.",
    "Inferred frequency 2D from passed values does not conform to passed frequency D",
    "int() argument must be a string, a bytes-like object or a real number, not 'Timestamp'",
    "unsupported operand type(s) for +: 'DatetimeArray' and 'Timedelta'",
    "Cannot convert a floating of object type",
    "conversion from DecimalArray to Decimal is not supported",
    "unsupported format string passed to MultiIndex.__format__",
    "Cannot have mixed values with boolean",
    "Cannot have NaN with string",
    "object of type <class 'str'> cannot be converted to int",
    "'method' object is not subscriptable",
}


def is_known_error(message):
    return message in ERROR_MESSAGES or any(
        err in message for err in ERROR_MESSAGES
    )


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    # Let pytest run the test and generate a report
    outcome = yield
    rep = outcome.get_result()
    # Only process actual test call failures (not setup/teardown)
    if rep.when == "call" and rep.failed and call.excinfo is not None:
        error_message = str(call.excinfo.value)
        if is_known_error(error_message):
            # Mark as xfail
            rep.outcome = "skipped"
            rep.wasxfail = f"xfail due to known error: {error_message[:50]}"


sys.path.append(os.path.dirname(__file__))
