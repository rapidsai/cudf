# Copyright (c) 2022-2023, NVIDIA CORPORATION.
import contextlib
import doctest
import inspect
import io
import itertools
import os

import numpy as np
import pytest

import cudf

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")

# modules that will be searched for doctests
tests = [cudf, cudf.core.groupby]


def _name_in_all(parent, name):
    return name in getattr(parent, "__all__", [])


def _is_public_name(parent, name):
    return not name.startswith("_")


def _find_doctests_in_obj(obj, finder=None, criteria=None):
    """Find all doctests in an object.

    Parameters
    ----------
    obj : module or class
        The object to search for docstring examples.
    finder : doctest.DocTestFinder, optional
        The DocTestFinder object to use. If not provided, a DocTestFinder is
        constructed.
    criteria : callable, optional
        Callable indicating whether to recurse over members of the provided
        object. If not provided, names not defined in the object's ``__all__``
        property are ignored.

    Yields
    ------
    doctest.DocTest
        The next doctest found in the object.
    """
    if finder is None:
        finder = doctest.DocTestFinder()
    if criteria is None:
        criteria = _name_in_all
    for docstring in finder.find(obj):
        if docstring.examples:
            yield docstring
    for name, member in inspect.getmembers(obj):
        # Only recurse over members matching the criteria
        if not criteria(obj, name):
            continue
        # Recurse over the public API of modules (objects defined in the
        # module's __all__)
        if inspect.ismodule(member):
            yield from _find_doctests_in_obj(
                member, finder, criteria=_name_in_all
            )
        # Recurse over the public API of classes (attributes not prefixed with
        # an underscore)
        if inspect.isclass(member):
            yield from _find_doctests_in_obj(
                member, finder, criteria=_is_public_name
            )


class TestDoctests:
    @pytest.fixture(autouse=True)
    def chdir_to_tmp_path(cls, tmp_path):
        # Some doctests generate files, so this fixture runs the tests in a
        # temporary directory.
        original_directory = os.getcwd()
        os.chdir(tmp_path)
        yield
        os.chdir(original_directory)

    @pytest.mark.parametrize(
        "docstring",
        itertools.chain(*[_find_doctests_in_obj(mod) for mod in tests]),
        ids=lambda docstring: docstring.name,
    )
    def test_docstring(self, docstring):
        # We ignore differences in whitespace in the doctest output, and enable
        # the use of an ellipsis "..." to match any string in the doctest
        # output. An ellipsis is useful for, e.g., memory addresses or
        # imprecise floating point values.
        optionflags = doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
        runner = doctest.DocTestRunner(optionflags=optionflags)

        # These global names are pre-defined and can be used in doctests
        # without first importing them.
        globals = dict(
            cudf=cudf,
            np=np,
        )
        docstring.globs = globals

        # Capture stdout and include failing outputs in the traceback.
        doctest_stdout = io.StringIO()
        with contextlib.redirect_stdout(doctest_stdout):
            runner.run(docstring)
            results = runner.summarize()
        assert not results.failed, (
            f"{results.failed} of {results.attempted} doctests failed for "
            f"{docstring.name}:\n{doctest_stdout.getvalue()}"
        )
