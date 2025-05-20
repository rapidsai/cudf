# Copyright (c) 2022-2025, NVIDIA CORPORATION.

"""Common utilities for fixture creation and benchmarking."""

import inspect
import re
import textwrap
from collections.abc import MutableSet
from itertools import groupby
from numbers import Real

import pytest_cases
from config import NUM_COLS, NUM_ROWS, cudf, cupy


def make_gather_map(len_gather_map: Real, len_column: Real, how: str):
    """Create a gather map based on "how" you'd like to gather from input.
    - sequence: gather the first `len_gather_map` rows, the first thread
                collects the first element
    - reverse:  gather the last `len_gather_map` rows, the first thread
                collects the last element
    - random:   create a pseudorandom gather map

    `len_gather_map`, `len_column` gets rounded to integer.
    """
    len_gather_map = round(len_gather_map)
    len_column = round(len_column)

    rstate = cupy.random.RandomState(seed=0)
    if how == "sequence":
        return cudf.Series(cupy.arange(0, len_gather_map))
    elif how == "reverse":
        return cudf.Series(
            cupy.arange(len_column - 1, len_column - len_gather_map - 1, -1)
        )
    elif how == "random":
        return cudf.Series(rstate.randint(0, len_column, len_gather_map))


def make_boolean_mask_column(size):
    rstate = cupy.random.RandomState(seed=0)
    return cudf.core.column.as_column(rstate.randint(0, 2, size).astype(bool))


def benchmark_with_object(
    cls, *, dtype="int", nulls=None, cols=None, rows=None
):
    """Pass "standard" cudf fixtures to functions without renaming parameters.

    The fixture generation logic in conftest.py provides a plethora of useful
    fixtures to allow developers to easily select an appropriate cross-section
    of the space of objects to apply a particular benchmark to. However, the
    usage of these fixtures is cumbersome because creating them in a principled
    fashion results in long names and very specific naming schemes. This
    decorator abstracts that naming logic away from the developer, allowing
    them to instead focus on defining the fixture semantically by describing
    its properties.

    Parameters
    ----------
    cls : Union[str, Type]
        The class of object to test. May either be specified as the type
        itself, or using the name (as a string). If a string, the case is
        irrelevant as the string will be converted to all lowercase.
    dtype : Union[str, Iterable[str]], default 'int'
        The dtype or set of dtypes to use.
    nulls : Optional[bool], default None
        Whether to test nullable or non-nullable data. If None, both nullable
        and non-nullable data are included.
    cols : Optional[int], None
        The number of columns. Only valid if cls == 'dataframe'. If None, use
        all possible numbers of columns. Specifying multiple values is
        unsupported.
    rows : Optional[int], None
        The number of rows. If None, use all possible numbers of rows.
        Specifying multiple values is unsupported.

    Raises
    ------
    AssertionError
        If any of the parameters do not correspond to extant fixtures.

    Examples
    --------
    # Note: As an internal function, this example is not meant for doctesting.

    @benchmark_with_object("dataframe", dtype="int", nulls=False)
    def bench_columns(benchmark, df):
        benchmark(df.columns)
    """
    if inspect.isclass(cls):
        cls = cls.__name__
    cls = cls.lower()

    supported_classes = (
        "column",
        "series",
        "index",
        "dataframe",
        "indexedframe",
        "frame_or_index",
    )
    assert cls in supported_classes, (
        f"cls {cls} is invalid, choose from {', '.join(supported_classes)}"
    )

    if not isinstance(dtype, list):
        dtype = [dtype]
    assert all(dt in column_generators for dt in dtype), (
        f"The only supported dtypes are {', '.join(column_generators)}"
    )

    dtype_str = "_dtype_" + "_or_".join(dtype)

    null_str = ""
    if nulls is not None:
        null_str = f"_nulls_{nulls}".lower()

    col_str = ""
    if cols is not None:
        assert cols in NUM_COLS, (
            f"You have requested a DataFrame with {cols} columns but fixtures "
            f"only exist for the values {', '.join(NUM_COLS)}"
        )
        col_str = f"_cols_{cols}"

    row_str = ""
    if rows is not None:
        assert rows in NUM_ROWS, (
            f"You have requested a {cls} with {rows} rows but fixtures "
            f"only exist for the values {', '.join(NUM_ROWS)}"
        )
        row_str = f"_rows_{rows}"

    fixture_name = f"{cls}{dtype_str}{null_str}{col_str}{row_str}"

    def deco(bm):
        # pytest's test collection process relies on parsing the globals dict
        # to find test functions and identify their parameters for the purpose
        # of fixtures and parameters. Therefore, the primary purpose of this
        # decorator is to define a new benchmark function with a signature
        # identical to that of the decorated benchmark except with the user's
        # fixture name replaced by the true fixture name based on the arguments
        # to benchmark_with_object.
        parameters = inspect.signature(bm).parameters

        # Note: This logic assumes that any benchmark using this fixture has at
        # least two parameters since they must be using both the
        # pytest-benchmark `benchmark` fixture and the cudf object.
        params_str = ", ".join(f"{p}" for p in parameters if p != cls)
        arg_str = ", ".join(f"{p}={p}" for p in parameters if p != cls)

        if params_str:
            params_str += ", "
        if arg_str:
            arg_str += ", "

        params_str += f"{fixture_name}"
        arg_str += f"{cls}={fixture_name}"

        src = textwrap.dedent(
            f"""
            import makefun
            @makefun.wraps(
                bm,
                remove_args=("{cls}",),
                prepend_args=("{fixture_name}",)
            )
            def wrapped_bm({params_str}):
                return bm({arg_str})
            """
        )
        globals_ = {"bm": bm}
        exec(src, globals_)
        wrapped_bm = globals_["wrapped_bm"]
        # In case marks were applied to the original benchmark, copy them over.
        if marks := getattr(bm, "pytestmark", None):
            wrapped_bm.pytestmark = marks
        wrapped_bm.place_as = bm
        return wrapped_bm

    return deco


class OrderedSet(MutableSet):
    """A minimal OrderedSet implementation built on a dict.

    This implementation exploits the fact that dicts are ordered as of Python
    3.7. It is not intended to be performant, so only the minimal set of
    methods are implemented. We need this class to ensure that fixture names
    are constructed deterministically, otherwise pytest-xdist will complain if
    different threads have seemingly different tests.
    """

    def __init__(self, args=None):
        args = args or []
        self._data = {value: None for value in args}

    def __contains__(self, key):
        return key in self._data

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        # Helpful for debugging.
        data = ", ".join(str(i) for i in self._data)
        return f"{self.__class__.__name__}({data})"

    def add(self, value):
        self._data[value] = None

    def discard(self, value):
        self._data.pop(value, None)


def make_fixture(name, func, globals_, fixtures):
    """Create a named fixture in `globals_` and save its name in `fixtures`.

    https://github.com/pytest-dev/pytest/issues/2424#issuecomment-333387206
    explains why this hack is necessary. Essentially, dynamically generated
    fixtures must exist in globals() to be found by pytest.
    """
    globals_[name] = pytest_cases.fixture(name=name)(func)
    fixtures.add(name)


def collapse_fixtures(fixtures, pattern, repl, globals_, idfunc=None):
    """Create unions of fixtures based on specific name mappings.

    `fixtures` are grouped into unions according the regex replacement
    `re.sub(pattern, repl)` and placed into `new_fixtures`.
    """

    def collapser(n):
        return re.sub(pattern, repl, n)

    # Note: sorted creates a new list, not a view, so it's OK to modify the
    # list of fixtures while iterating over the sorted result.
    for name, group in groupby(sorted(fixtures, key=collapser), key=collapser):
        group = list(group)
        if len(group) > 1 and name not in fixtures:
            pytest_cases.fixture_union(name=name, fixtures=group, ids=idfunc)
            # Need to assign back to the parent scope's globals.
            globals_[name] = globals()[name]
            fixtures.add(name)


# A dictionary of callables that create a column of a specified length
random_state = cupy.random.RandomState(42)
column_generators = {
    "int": (lambda nr: random_state.randint(low=0, high=100, size=nr)),
    "float": (lambda nr: random_state.rand(nr)),
}
