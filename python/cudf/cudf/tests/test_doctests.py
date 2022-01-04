import doctest
import inspect
import os
from contextlib import AbstractContextManager

import numpy as np
import pytest

import cudf


def _name_in_all(parent, name, member):
    return name in getattr(parent, "__all__", [])


def _is_public_name(parent, name, member):
    return not name.startswith("_")


def _find_docstrings_in_obj(finder, obj, criteria=None):
    for docstring in finder.find(obj):
        if docstring.examples:
            yield docstring
    for name, member in inspect.getmembers(obj):
        # Filter out non-matching objects with criteria
        if criteria is not None and not criteria(obj, name, member):
            continue
        # Recurse over the public API of modules (objects defined in __all__)
        if inspect.ismodule(member):
            yield from _find_docstrings_in_obj(
                finder, member, criteria=_name_in_all
            )
        # Recurse over the public API of classes (attributes not prefixed with
        # an underscore)
        if inspect.isclass(member):
            yield from _find_docstrings_in_obj(
                finder, member, criteria=_is_public_name
            )


def _fetch_doctests():
    finder = doctest.DocTestFinder()
    yield from _find_docstrings_in_obj(finder, cudf, criteria=_name_in_all)


class _chdir(AbstractContextManager):
    """Non thread-safe context manager to change the current working directory.

    Implementation copied from Python's contextlib.chdir, implemented in
    October 2021. This is not yet released but can be replaced with
    contextlib.chdir in the future.
    """

    def __init__(self, path):
        self.path = path
        self._old_cwd = []

    def __enter__(self):
        self._old_cwd.append(os.getcwd())
        os.chdir(self.path)

    def __exit__(self, *excinfo):
        os.chdir(self._old_cwd.pop())


class TestDoctests:
    @pytest.mark.parametrize(
        "docstring", _fetch_doctests(), ids=lambda docstring: docstring.name
    )
    def test_docstring(self, docstring, tmp_path):
        optionflags = doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
        runner = doctest.DocTestRunner(optionflags=optionflags)
        globs = dict(cudf=cudf, np=np,)
        docstring.globs = globs
        with _chdir(tmp_path):
            runner.run(docstring)
        results = runner.summarize()
        if results.failed:
            raise AssertionError(results)
