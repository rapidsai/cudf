# Copyright (c) 2022, NVIDIA CORPORATION.

import inspect
import textwrap

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing._utils import assert_eq


def _is_cudf(lib):
    # TODO: Is it safe to use `if lib is cudf`? I think there are some cases
    # where the imports of a package in different modules may not be considered
    # identical by `is` so we have to check the name as below, but this is
    # probably worth verifying.
    return lib.__name__ == "cudf"


# The parameter name used for pandas/cudf in test functions.
_LIB_PARAM_NAME = "lib"


def pandas_comparison_test(*args, cudf_objects=None, assert_func=assert_eq):
    def deco(test):
        """Run a test function with cudf and pandas and ensure equal results.

        This decorator generates a new function that looks identical to the
        decorated function but calls the original function twice, once each for
        pandas and cudf and asserts that the two results are equal.
        """
        # Handle parameters (fixtures or parametrize fixtures, it doesn't
        # matter) that are cudf objects that need to be converted to pandas.
        nonlocal cudf_objects
        if not cudf_objects:
            cudf_objects = []
        elif isinstance(cudf_objects, str):
            cudf_objects = [cudf_objects]

        parameters = inspect.signature(test).parameters
        params_str = ", ".join(
            f"{p}" for p in parameters if p != _LIB_PARAM_NAME
        )

        arg_str = ", ".join(
            f"{p}={p}"
            for p in parameters
            if (p != _LIB_PARAM_NAME and p not in cudf_objects)
        )

        if _LIB_PARAM_NAME in parameters:
            if arg_str:
                arg_str += ", "

            cudf_arg_str = arg_str + f"{_LIB_PARAM_NAME}=cudf"
            pandas_arg_str = arg_str + f"{_LIB_PARAM_NAME}=pandas"
        else:
            cudf_arg_str = pandas_arg_str = arg_str

        if cudf_objects:
            if cudf_arg_str:
                cudf_arg_str += ", "
            if pandas_arg_str:
                pandas_arg_str += ", "

            cudf_arg_str += ", ".join(
                f"{fixture}={fixture}" for fixture in cudf_objects
            )
            pandas_arg_str += ", ".join(
                f"{fixture}={fixture}.to_pandas()" for fixture in cudf_objects
            )

        src = textwrap.dedent(
            f"""
            import pandas
            import cudf
            import makefun
            @makefun.wraps(
                test,
                remove_args=remove_args,
            )
            def wrapped_test({params_str}):
                cudf_output = test({cudf_arg_str})
                pandas_output = test({pandas_arg_str})
                assert_func(pandas_output, cudf_output)
            """
        )
        globals_ = {
            "test": test,
            "assert_func": assert_func,
            "remove_args": (f"{_LIB_PARAM_NAME}",)
            if _LIB_PARAM_NAME in parameters
            else None,
        }
        exec(src, globals_)
        wrapped_test = globals_["wrapped_test"]
        # In case marks were applied to the original benchmark, copy them over.
        if marks := getattr(test, "pytestmark", None):
            wrapped_test.pytestmark = marks
        return wrapped_test

    if args:
        if len(args) == 1:
            # Was called directly with a test.
            return deco(args[0])
        raise ValueError("This decorator only supports keyword arguments.")
    return deco


@pandas_comparison_test
def test_init(lib):
    data = [
        (5, "cats", "jump", np.nan),
        (2, "dogs", "dig", 7.5),
        (3, "cows", "moo", -2.1, "occasionally"),
    ]
    return lib.DataFrame(data)


@pandas_comparison_test()
def test_init2(lib):
    data = [
        (5, "cats", "jump", np.nan),
        (2, "dogs", "dig", 7.5),
        (3, "cows", "moo", -2.1, "occasionally"),
    ]
    return lib.DataFrame(data)


@pytest.fixture
def dt():
    return [
        (5, "cats", "jump", np.nan),
        (2, "dogs", "dig", 7.5),
        (3, "cows", "moo", -2.1, "occasionally"),
    ]


@pandas_comparison_test
def test_init4(lib, dt):
    return lib.DataFrame(dt)


@pandas_comparison_test
def test_init7(dt, lib):
    return lib.DataFrame(dt)


@pandas_comparison_test
@pytest.mark.parametrize(
    "data",
    [
        [
            (5, "cats", "jump", np.nan),
            (2, "dogs", "dig", 7.5),
            (3, "cows", "moo", -2.1, "occasionally"),
        ]
    ],
)
def test_init5(lib, data):
    return lib.DataFrame(data)


@pytest.mark.parametrize(
    "data",
    [
        [
            (5, "cats", "jump", np.nan),
            (2, "dogs", "dig", 7.5),
            (3, "cows", "moo", -2.1, "occasionally"),
        ]
    ],
)
@pandas_comparison_test
def test_init6(lib, data):
    return lib.DataFrame(data)


@pytest.mark.parametrize(
    "df",
    [
        pd.DataFrame(
            [
                (5, "cats", "jump", np.nan),
                (2, "dogs", "dig", 7.5),
                (3, "cows", "moo", -2.1, "occasionally"),
            ]
        )
    ],
)
@pandas_comparison_test
def test_from_pandas(lib, df):
    return lib.from_pandas(df) if _is_cudf(lib) else df


def custom_assert(expected, got):
    cudf.testing.assert_frame_equal(cudf.from_pandas(expected), got)


@pytest.mark.parametrize(
    "df",
    [
        pd.DataFrame(
            [
                (5, "cats", "jump", np.nan),
                (2, "dogs", "dig", 7.5),
                (3, "cows", "moo", -2.1, "occasionally"),
            ]
        )
    ],
)
@pandas_comparison_test(assert_func=custom_assert)
def test_from_pandas_custom_assert(lib, df):
    return lib.from_pandas(df) if _is_cudf(lib) else df


@pytest.fixture
def df():
    return cudf.DataFrame(
        [
            (5, "cats", "jump", np.nan),
            (2, "dogs", "dig", 7.5),
            (3, "cows", "moo", -2.1, "occasionally"),
        ]
    )


@pandas_comparison_test(cudf_objects="df")
def test_fixture_concat(lib, df):
    return lib.concat([df, df])


@pytest.mark.parametrize(
    "df",
    [
        cudf.DataFrame(
            [
                (5, "cats", "jump", np.nan),
                (2, "dogs", "dig", 7.5),
                (3, "cows", "moo", -2.1, "occasionally"),
            ]
        )
    ],
)
@pandas_comparison_test(cudf_objects="df")
def test_param_concat(lib, df):
    return lib.concat([df, df])


def set_element(frame):
    """Change the first element of frame."""
    # Handle default values to replace with for different dtypes. We provide
    # two options to ensure that at least one of them is different from the
    # original value of the element we plan to overwrite.
    defaults_by_kind = {
        "b": (True, False),
        "i": (0, 1),
        "u": (0, 1),
        "O": ("a", "b"),
    }

    if isinstance(frame, (cudf.Series, pd.Series)):
        try:
            defaults = defaults_by_kind[frame.dtype.kind]
        except KeyError:
            raise TypeError("Provided frame has an unsupported dtype.")

        default = defaults[0] if frame.iloc[0] != defaults[0] else defaults[1]
        frame.iloc[0] = default
    elif isinstance(frame, (cudf.DataFrame, pd.DataFrame)):
        try:
            defaults = defaults_by_kind[frame.iloc[:, 0].dtype.kind]
        except KeyError:
            raise TypeError("Provided frame has an unsupported dtype.")
        default = (
            defaults[0] if frame.iloc[0, 0] != defaults[0] else defaults[1]
        )
        frame.iloc[0, 0] = default
    else:
        raise TypeError(
            f"Object {frame} of type {type(frame)} is unsupported."
        )


def pandas_copy_semantics_comparison_test(cudf_object, modify=set_element):
    def deco(test):
        """Verify that cudf and pandas methods have the same semantics.

        This decorator generates a new function that looks identical to the
        decorated function but calls the original function twice, once each for
        pandas and cudf. The function is assumed to take a single input
        """
        # These tests must have exactly one parameter that is a cudf object.
        nonlocal cudf_object

        parameters = inspect.signature(test).parameters
        params_str = ", ".join(f"{p}" for p in parameters)
        arg_str = ", ".join(f"{p}={p}" for p in parameters if p != cudf_object)

        if arg_str:
            arg_str += ", "

        cudf_arg_str = arg_str + f"{cudf_object}={cudf_object}"
        pandas_arg_str = arg_str + f"{cudf_object}=pandas_object"

        src = textwrap.dedent(
            f"""
            import pandas
            import cudf
            import makefun
            @makefun.wraps(
                test,
                remove_args=remove_args,
            )
            def wrapped_test({params_str}):
                cudf_object_orig = {cudf_object}.copy()

                pandas_object = {cudf_object}.to_pandas()
                pandas_object_orig = pandas_object.copy()

                cudf_output = test({cudf_arg_str})
                pandas_output = test({pandas_arg_str})

                modify(cudf_output)
                modify(pandas_output)

                cudf_modified = {cudf_object}.equals(cudf_object_orig)
                pandas_modified = pandas_object.equals(pandas_object_orig)
                assert cudf_modified == pandas_modified, (
                    f"The cudf object was{{' not' if cudf_modified else ''}} "
                    "modified, while pandas object was"
                    f"{{' not' if pandas_modified else ''}}."
                )
            """
        )
        globals_ = {
            "test": test,
            "remove_args": (f"{_LIB_PARAM_NAME}",)
            if _LIB_PARAM_NAME in parameters
            else None,
            "cudf_object": cudf_object,
            "modify": modify,
        }
        exec(src, globals_)
        wrapped_test = globals_["wrapped_test"]
        # In case marks were applied to the original benchmark, copy them over.
        if marks := getattr(test, "pytestmark", None):
            wrapped_test.pytestmark = marks
        return wrapped_test

    return deco


# TODO: Instead of requiring the user to pass `cudf_object`, we could require
# the cudf object (fixture or parameter) to be the first argument. The current
# approach seems more explicit, but perhaps we would prefer the other, less
# verbose approach?
@pandas_copy_semantics_comparison_test(cudf_object="df")
def test_df_head(df):
    return df.head()
