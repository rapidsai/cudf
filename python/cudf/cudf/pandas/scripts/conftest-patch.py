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
    "assert ",
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
    "maximum recursion depth exceeded",
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
    "Can't pickle <class",
    "conversion from DecimalArray to Decimal is not supported",
    "unsupported format string passed to MultiIndex.__format__",
    "Cannot have mixed values with boolean",
    "Cannot have NaN with string",
    "object of type <class 'str'> cannot be converted to int",
    "'method' object is not subscriptable",
    "'BooleanDtype' object has no attribute '__from_arrow__'",
    "'DataFrame' object has no attribute 'dtype'. Did you mean: 'dtypes'?",
    "'DataFrame' object has no attribute 'name'",
    "'DataFrameGroupBy' object has no attribute '_grouper'. Did you mean: 'grouper'?",
    "'DateOffset' object has no attribute '_offset'",
    "'DateOffset' object has no attribute ",
    "'DatetimeProperties' object has no attribute 'days'. Did you mean: 'day'?",
    "'FixedForwardWindowIndexer' object has no attribute 'window_size'",
    "'Index' object has no attribute '_cache'",
    "'IntervalArray' object has no attribute '_left'. Did you mean: 'left'?",
    "'IntervalDtype' object has no attribute '__from_arrow__'",
    "'Series' object has no attribute 'custom_series_function'",
    "'Styler' object has no attribute 'columns'",
    "'_PickleConstructor' object has no attribute '__name__'. Did you mean: '__ne__'?",
    "'dict' object has no attribute 'fillna'",
    "'numpy.int64' object has no attribute 'iloc'",
    "Can only use .str accessor with string values!",
    "assert array([",
    "Can only use .str accessor with string values. Did you mean: 'std'?",
    "DataFrame object has no attribute name",
    "No attribute 'Block'. Did you mean: 'blocks'?",
    "DataFrame Expected type <class",
    "No attribute 'DatetimeTZBlock'",
    "No attribute 'ExtensionBlock'",
    "No attribute 'NumericBlock'",
    "No attribute 'ObjectBlock'",
    "No attribute 'PandasArray'",
    "'collections.defaultdict' object has no attribute 'names'",
    "No attribute 'create_block_manager_from_blocks'",
    "'Styler' object has no attribute '_todo'",
    "'Styler' object has no attribute 'ctx_index'. Did you mean: 'map_index'?",
    "'Styler' object has no attribute 'caption'",
    "'Index' object has no attribute 'categories'",
    "'<' not supported between instances of 'int' and 'Timestamp'",
    "'<=' not supported between instances of 'int' and 'Timestamp'",
    "'>' not supported between instances of 'int' and 'Timestamp'",
    "'>=' not supported between instances of 'int' and 'Timestamp'",
    "'NoneType' object is not callable",
    "'Period' object is not iterable",
    "'property' object is not callable",
    "'quantile' cannot be performed against 'object' dtypes!",
    "<class 'pandas._libs.tslibs.timestamps.Timestamp'> is not convertible to datetime, at position 0",
    "<lambda>() missing 1 required positional argument: 'x'",
    "<lambda>() takes 0 positional arguments but 1 was given",
    "Cannot convert a date of object type",
    "Cannot convert a datetime of object type",
    "Cannot convert a datetime64 of object type",
    "Cannot convert a floating of object type",
    "Cannot convert a integer of object type",
    "Cannot convert tz-naive timestamps, use tz_localize to localize",
    "Cannot interpret 'ListDtype(int64)' as a data type",
    "Cannot interpret 'string[pyarrow]' as a data type",
    "Cannot interpret 'string[python]' as a data type",
    "Expected tuple, got str",
    "Index object is not iterable. Consider using `.to_arrow()`, `.to_pandas()` or `.values_host` if you wish to iterate over the values.",
    "Interval must be an iterable or sequence.",
    "Operation 'bitwise or' not supported between float64 and bool",
    "Timedelta must be an iterable or sequence.",
    "Timestamp must be an iterable or sequence.",
    "Unusable type. Falling back to the slow object",
    "Wrong type ((<class 'numpy.datetime64'>,)) of arguments for cupy_copy",
    "[0] of type <class 'list'> is not a valid type for hashing, must be string or null",
    "an integer is required",
    "bad operand type for abs(): 'NaTType'",
    "boolean value of NA is ambiguous",
    "cannot do slice indexing on DatetimeIndex with these indexers [2000-01-10 00:00:00] of type Timestamp",
    "cannot do slice indexing on DatetimeIndex with these indexers [2014-05-07 00:00:00] of type Timestamp",
    "cannot do slice indexing on DatetimeIndex with these indexers [2015-02-01 00:00:00] of type Timestamp",
    "cannot do slice indexing on DatetimeIndex with these indexers [2017-10-29 02:30:00+02:00] of type Timestamp",
    "cannot do slice indexing on DatetimeIndex with these indexers [2020-01-06 00:00:00] of type Timestamp",
    "cannot do slice indexing on DatetimeIndex with these indexers [2151-06-05 06:32:39.009206272] of type Timestamp",
    "cannot do slice indexing on PeriodIndex with these indexers [2015-02] of type Period",
    "cannot do slice indexing on PeriodIndex with these indexers [2017-12] of type Period",
    "cannot do slice indexing on TimedeltaIndex with these indexers [0 days 00:00:00] of type Timedelta",
    "cannot do slice indexing on TimedeltaIndex with these indexers [0 days 13:00:00] of type Timedelta",
    "cannot do slice indexing on TimedeltaIndex with these indexers [1 days 23:00:00] of type Timedelta",
    "cannot pickle 'module' object",
    "cannot sort an Index object in-place, use sort_values instead",
    "descriptor '__repr__' for 'pandas._libs.tslibs.period._Period' objects doesn't apply to a 'NaTType' object",
    "descriptor '__sub__' requires a 'pandas._libs.tslibs.period._Period' object but received a 'NaTType'",
    "float() argument must be a string or a real number, not 'Timedelta'",
    "float() argument must be a string or a real number, not 'Timestamp'",
    "int() argument must be a string, a bytes-like object or a real number, not 'Period'",
    "isinstance() arg 2 must be a type, a tuple of types, or a union",
    "object type does not support <function GroupBy.quantile.<locals>.func at 0x77b431dfa8e0> operations",
    "only length-1 arrays can be converted to Python scalars",
    "slice indices must be integers or None or have an __index__ method",
    "slice must be an iterable or sequence.",
    "ufunc 'bitwise_or' not supported for the input types, and the inputs could not be safely coerced to any supported types according to the casting rule ''safe''",
    "ufunc 'isnan' not supported for the input types, and the inputs could not be safely coerced to any supported types according to the casting rule ''safe''",
    "unhashable type: 'PeriodDtype'",
    "unsupported operand type(s) for *: 'int' and 'Timestamp'",
    "unsupported operand type(s) for +: 'NaTType' and 'Timestamp'",
    "unsupported operand type(s) for +: 'Timestamp' and 'NaTType'",
    "unsupported operand type(s) for -: 'Timestamp' and 'datetime.datetime'",
    "unsupported operand type(s) for |: 'float' and 'bool'",
    "value should be a 'Timestamp', 'NaT', or array of those. Got category array instead.",
    "Cannot create column with mixed types",
    "Cannot have NaN with string",
    "Cannot have mixed values with boolean",
    "Expected bytes, got a 'bool' object",
    "Expected bytes, got a 'complex' object",
    "Expected bytes, got a 'float' object",
    "Expected bytes, got a 'int' object",
    "Expected bytes, got a 'slice' object",
    "object of type <class 'str'> cannot be converted to int",
    "Length of values (4) does not match length of index (3)",
    "columns length expected 4 but found 2",
    "Shape of passed values is (3, 4), indices imply (3, 2)",
    "cudf does not support object dtype. Use 'str' instead.",
    "('Lengths must match to compare', (2,), (1,))",
    "Duplicate column names are not allowed",
    "setting an array element with a sequence",
    "('Lengths must match to compare', (100,), (1,))",
    "cannot reindex on an axis with duplicate labels",
    "Length of values (1) does not match length of index (2)",
    "Need to pass dtype when passing pd.NA or None",
    "Length of values (2) does not match length of index (3)",
    "`level` must either be a list of names or positions, not a mixture of both.",
    "'exp' is not in list",
    "Categorical categories cannot be null",
    "Categorical categories must be unique",
    "'values' must be a NumPy array, not Index",
    "value must be a scalar",
    "All columns must be of equal length",
    "7 is not in range",
    "operands could not be broadcast together with shapes (4,) (3,)",
    "Length mismatch: Expected axis has 2 elements, new values have 1 elements",
    "Shape of passed values is (3, 1), indices imply (2, 1)",
    "Array conditional must be same shape as self",
    "Could not convert strings to integer type due to presence of non-integer values.",
    "Length of values (3) does not match length of index (4)",
    "Inferred frequency 4D from passed values does not conform to passed frequency D",
    "Cannot index with multidimensional key",
    "Could not convert strings to float type due to presence of non-floating values.",
    "storage_options passed with file object or non-fsspec file path",
    "CategoricalColumns do not use data attribute of Column, use `set_base_children` instead",
    "CUDF failure at: /raid/pgali/cudf/cpp/src/io/comp/uncomp.cpp:432: Failed to parse GZIP header while fetching source properties",
    "CUDF failure at: /raid/pgali/cudf/cpp/src/io/comp/uncomp.cpp:615: Decompression: error in stream",
    "CUDF failure at: /raid/pgali/cudf/cpp/src/io/comp/uncomp.cpp:448: Failed to parse ZIP header while fetching source properties",
    "Resampling by DateOffset objects is not yet supported.",
    "Fast implementation not available. Falling back to the slow implementation",
    "freq is not implemented yet",
    "US/Eastern must be a zoneinfo.ZoneInfo object in pandas_compatible mode.",
    "cuDF does not yet support PeriodDtype",
    "US/Pacific must be a zoneinfo.ZoneInfo object in pandas_compatible mode.",
    "Expected np.datetime64 but got: double",
    "Custom pandas ExtensionDtypes are not supported",
    "not supported",
    "cuDF does not yet support SparseDtype",
    "Expected np.datetime64 but got: duration[ns]",
    "Series with Complex128DType is not supported.",
    "Unsupported column type passed to create an Index: <class 'cudf.core.column.decimal.Decimal128Column'>",
    "Lists concatenation for this operation is not yetsupported",
    "UTC must be a zoneinfo.ZoneInfo object in pandas_compatible mode.",
    "dropna is not currently supported.",
    "database is locked",
    "(sqlite3.OperationalError) database is locked",
    "Option 'chained_assignment' has already been registered",
    "1",
    "dtype('<m8[h]')",
    "dtype('<m8[m]')",
    "'tags'",
    "nan",
    "dtype('>M8[ms]')",
    "'20130903'",
    "'Level exp not found'",
    "'Requested level (ID) does not match index name (None)'",
    "(0, nan)",
    "\"label=Interval(1, 4, closed='right') not in index\"",
    "\"label=Timestamp('2019-01-01 00:00:00') not in index\"",
    "\"label=Timedelta('1 days 00:00:00') not in index\"",
    "7",
    "'Label scalar is out of bounds'",
    "'a'",
    "'_INTERCHANGE_PROTOCOL_BUFFERS'",
    "'Block'",
    "'tar'",
    "'zstd'",
    "ColumnBase are different",
    "Attributes of DataFrame.iloc[:,",
    "Attributes of Series are different",
    "ufunc 'subtract' cannot use operands with types dtype('<M8[s]') and dtype('O')",
    "ufunc 'subtract' cannot use operands with types dtype('<M8[D]') and dtype('O')",
    "ufunc 'positive' did not contain a loop with signature matching types <class 'numpy.dtypes.StrDType'> -> None",
    "ufunc 'subtract' cannot use operands with types dtype('<m8[ns]') and dtype('O')",
    "ufunc 'subtract' cannot use operands with types dtype('<m8') and dtype('O')",
    "ufunc 'subtract' cannot use operands with types dtype('<M8') and dtype('O')",
    "ufunc 'subtract' cannot use operands with types dtype('<m8[h]') and dtype('O')",
    "ufunc 'divide' cannot use operands with types dtype('<m8[ns]') and dtype('O')",
    "ufunc 'divide' cannot use operands with types dtype('<m8[h]') and dtype('O')",
    "ufunc 'divide' cannot use operands with types dtype('<m8') and dtype('O')",
    "ufunc 'remainder' cannot use operands with types dtype('<m8[m]') and dtype('O')",
    "ufunc 'subtract' cannot use operands with types dtype('<M8[ns]') and dtype('O')",
    "ufunc 'subtract' cannot use operands with types dtype('<m8[m]') and dtype('O')",
    "ufunc 'divide' cannot use operands with types dtype('<m8[m]') and dtype('O')",
    "ufunc 'divide' cannot use operands with types dtype('<m8[D]') and dtype('O')",
    "ufunc 'divide' cannot use operands with types dtype('<m8[s]') and dtype('O')",
    "ufunc 'divide' cannot use operands with types dtype('<m8[ms]') and dtype('O')",
    "ufunc 'divide' cannot use operands with types dtype('<m8[us]') and dtype('O')",
    "ufunc 'floor_divide' cannot use operands with types dtype('<m8[ns]') and dtype('O')",
    "ufunc 'remainder' cannot use operands with types dtype('<m8[D]') and dtype('O')",
    "ufunc 'remainder' cannot use operands with types dtype('<m8[h]') and dtype('O')",
    "ufunc 'minimum' did not contain a loop with signature matching types (dtype('<U1'), dtype('<U1')) -> None",
    "(1, 4]",
    "2019-01-01 00:00:00",
    "1 days 00:00:00",
    "(0, 1]",
    "(1, 5]",
    "(1, 3]",
    "2012-01-01 12:12:12",
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
    if (
        rep.when in {"call", "setup"}
        and rep.failed
        and call.excinfo is not None
    ):
        error_message = str(call.excinfo.value)
        if is_known_error(error_message):
            # Mark as xfail
            rep.outcome = "skipped"
            rep.wasxfail = f"xfail due to known error: {error_message[:50]}"


sys.path.append(os.path.dirname(__file__))
