# Copyright (c) 2022-2024, NVIDIA CORPORATION.

"""Defines pytest fixtures for all benchmarks.

Most fixtures defined in this file represent one of the primary classes in the
cuDF ecosystem such as DataFrame, Series, or Index. These fixtures may in turn
be broken up into two categories: base fixtures and fixture unions. Each base
fixture represents a specific type of object as well as certain of its
properties crucial for benchmarking. Specifically, fixtures must account for
the following different parameters:
    - Class of object (DataFrame, Series, Index)
    - Dtype
    - Nullability
    - Size (rows for all, rows/columns for DataFrame)

One such fixture is a series of nullable integer data. Given that we generally
want data across different sizes, we parametrize all fixtures across different
numbers of rows rather than generating separate fixtures for each different
possible number of rows. The number of columns is only relevant for DataFrame.

While this core set of fixtures means that any benchmark can be run for any
combination of these parameters, it also means that we would effectively have
to parametrize our benchmarks with many fixtures. Not only is parametrizing
tests with fixtures in this manner unsupported by pytest, it is also an
inelegant solution leading to cumbersome parameter lists that must be
maintained across all tests. Instead we make use of the
`pytest_cases <https://smarie.github.io/python-pytest-cases/>_` pytest plugin,
which supports the creation of fixture unions: fixtures that result from
combining other fixtures together. The result is a set of well-defined fixtures
that allow us to write benchmarks that naturally express the set of objects for
which they are valid, e.g. `def bench_sort_values(frame_or_index)`.

The generated fixtures are named according to the following convention:
`{classname}_dtype_{dtype}[_nulls_{true|false}][_cols_{num_cols}][_rows_{num_rows}]`
where classname is one of the following: index, series, dataframe,
indexedframe, frame, frame_or_index. Note that in the case of indexes, to match
Series/DataFrame we simply set `classname=index` and rely on the
`dtype_{dtype}` component to delineate which index class is actually in use.

In addition to the above fixtures, we also provide the following more
specialized fixtures:
    - rangeindex: Since RangeIndex always holds int64 data we cannot conflate
      it with index_dtype_int64 (a true Index with int64 dtype), and it
      cannot hold nulls. As a result, it is provided as a separate fixture.
"""

import os
import string
import sys

import pytest_cases

# TODO: Rather than doing this path hacking (including the sessionstart and
# sessionfinish hooks), we could just make the benchmarks a (sub)package to
# enable relative imports. A minor change to consider when these are ported
# into the main repo.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "common"))

# Turn off isort until we upgrade to 5.8.0
# https://github.com/pycqa/isort/issues/1594
from config import (  # noqa: W0611, E402, F401
    NUM_COLS,
    NUM_ROWS,
    collect_ignore,
    cudf,  # noqa: W0611, E402, F401
    pytest_collection_modifyitems,
    pytest_sessionfinish,
    pytest_sessionstart,
)
from utils import (  # noqa: E402
    OrderedSet,
    collapse_fixtures,
    column_generators,
    make_fixture,
)


@pytest_cases.fixture(params=[0, 1], ids=["AxisIndex", "AxisColumn"])
def axis(request):
    return request.param


# First generate all the base fixtures.
fixtures = OrderedSet()
for dtype, column_generator in column_generators.items():

    def make_dataframe(nr, nc, column_generator=column_generator):
        assert nc <= len(
            string.ascii_lowercase
        ), "make_dataframe only supports a maximum of 26 columns"
        return cudf.DataFrame(
            {
                f"{string.ascii_lowercase[i]}": column_generator(nr)
                for i in range(nc)
            }
        )

    for nr in NUM_ROWS:
        # TODO: pytest_cases.fixture doesn't appear to support lambdas where
        # pytest does. https://github.com/smarie/python-pytest-cases/issues/278
        # Once that is fixed we could use lambdas here.
        # TODO: pytest_cases has a bug where the first argument being a
        # defaulted kwarg e.g. (nr=nr, nc=nc) raises errors.
        # https://github.com/smarie/python-pytest-cases/issues/278
        # Once that is fixed we could remove all the extraneous `request`
        # fixtures in these fixtures.
        def series_nulls_false(
            request, nr=nr, column_generator=column_generator
        ):
            return cudf.Series(column_generator(nr))

        make_fixture(
            f"series_dtype_{dtype}_nulls_false_rows_{nr}",
            series_nulls_false,
            globals(),
            fixtures,
        )

        def series_nulls_true(
            request, nr=nr, column_generator=column_generator
        ):
            s = cudf.Series(column_generator(nr))
            s.iloc[::2] = None
            return s

        make_fixture(
            f"series_dtype_{dtype}_nulls_true_rows_{nr}",
            series_nulls_true,
            globals(),
            fixtures,
        )

        # For now, not bothering to include a nullable index fixture.
        def index_nulls_false(
            request, nr=nr, column_generator=column_generator
        ):
            return cudf.Index(column_generator(nr))

        make_fixture(
            f"index_dtype_{dtype}_nulls_false_rows_{nr}",
            index_nulls_false,
            globals(),
            fixtures,
        )

        for nc in NUM_COLS:

            def dataframe_nulls_false(
                request, nr=nr, nc=nc, make_dataframe=make_dataframe
            ):
                return make_dataframe(nr, nc)

            make_fixture(
                f"dataframe_dtype_{dtype}_nulls_false_cols_{nc}_rows_{nr}",
                dataframe_nulls_false,
                globals(),
                fixtures,
            )

            def dataframe_nulls_true(
                request, nr=nr, nc=nc, make_dataframe=make_dataframe
            ):
                df = make_dataframe(nr, nc)
                df.iloc[::2, :] = None
                return df

            make_fixture(
                f"dataframe_dtype_{dtype}_nulls_true_cols_{nc}_rows_{nr}",
                dataframe_nulls_true,
                globals(),
                fixtures,
            )


# We define some custom naming functions for use in the creation of fixture
# unions to create more readable test function names that don't contain the
# entire union, which quickly becomes intractably long.
def unique_union_id(val):
    return val.alternative_name


def default_union_id(val):
    return f"alt{val.get_alternative_idx()}"


# Label the first level differently from others since there's no redundancy.
idfunc = unique_union_id
num_new_fixtures = len(fixtures)

# Keep trying to merge existing fixtures until no new fixtures are added.
while num_new_fixtures > 0:
    num_fixtures = len(fixtures)

    # Note: If we start also introducing unions across dtypes, most likely
    # those will take the form `*int_and_float*` or similar since we won't want
    # to union _all_ dtypes. In that case, the regexes will need to use
    # suitable lookaheads etc to avoid infinite loops here.
    for pat, repl in [
        ("_nulls_(true|false)", ""),
        ("series|dataframe", "indexedframe"),
        ("indexedframe|index", "frame_or_index"),
        (r"_rows_\d+", ""),
        (r"_cols_\d+", ""),
    ]:
        collapse_fixtures(fixtures, pat, repl, globals(), idfunc)

    num_new_fixtures = len(fixtures) - num_fixtures
    # All subsequent levels get the same (collapsed) labels.
    idfunc = default_union_id


for dtype, column_generator in column_generators.items():
    # We have to manually add this one because we aren't including nullable
    # indexes but we want to be able to run some benchmarks on Series/DataFrame
    # that may or may not be nullable as well as Index objects.
    pytest_cases.fixture_union(
        name=f"frame_or_index_dtype_{dtype}",
        fixtures=(
            f"indexedframe_dtype_{dtype}",
            f"index_dtype_{dtype}_nulls_false",
        ),
        ids=["", f"index_dtype_{dtype}_nulls_false"],
    )


# TODO: Decide where to incorporate RangeIndex and MultiIndex fixtures.
@pytest_cases.fixture(params=NUM_ROWS)
def rangeindex(request):
    return cudf.RangeIndex(request.param)
