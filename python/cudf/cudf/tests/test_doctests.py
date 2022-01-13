import doctest
import inspect
import os

import numpy as np
import pytest

import cudf


def _name_in_all(parent, name):
    return name in getattr(parent, "__all__", [])


def _is_public_name(parent, name):
    return not name.startswith("_")


def _find_doctests_in_obj(finder, obj, criteria):
    """Find all doctests in an object.

    Parameters
    ----------
    finder : doctest.DocTestFinder
        The DocTestFinder object to use.
    obj : module or class
        The object to search for docstring examples.
    criteria : callable
        Callable indicating whether to recurse over members of the provided
        object.

    Yields
    ------
    doctest.DocTest
        The next doctest found in the object.
    """
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
                finder, member, criteria=_name_in_all
            )
        # Recurse over the public API of classes (attributes not prefixed with
        # an underscore)
        if inspect.isclass(member):
            yield from _find_doctests_in_obj(
                finder, member, criteria=_is_public_name
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
        _find_doctests_in_obj(
            finder=doctest.DocTestFinder(), obj=cudf, criteria=_name_in_all
        ),
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
        globals = dict(cudf=cudf, np=np,)
        docstring.globs = globals

        runner.run(docstring)
        results = runner.summarize()
        assert not results.failed, results
