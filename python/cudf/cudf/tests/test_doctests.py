import doctest
import inspect

import pytest

import cudf

# These classes and all subclasses will be doctested
doctested_classes = [
    "Frame",
    "BaseIndex",
]


def find_docstrings_in_module(finder, module):
    print("Finding in module", module.__name__)
    for docstring in finder.find(module):
        print("Finding in docstring", docstring.name, docstring.filename)
        if docstring.examples:
            yield docstring
    for name, member in inspect.getmembers(module):
        if name not in getattr(module, "__all__", []):
            if inspect.ismodule(member):
                print("SKIPPING MODULE", module.__name__, name)
            else:
                print("Skipping member", module.__name__, name)
            continue
        # print("Investigating", name)
        if inspect.ismodule(member):
            yield from find_docstrings_in_module(finder, member)


def fetch_doctests():
    finder = doctest.DocTestFinder()
    yield from find_docstrings_in_module(finder, cudf)


class TestDoctests:
    @pytest.mark.parametrize(
        "docstring", fetch_doctests(), ids=lambda docstring: docstring.name
    )
    def test_docstring(self, docstring):
        optionflags = doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
        runner = doctest.DocTestRunner(optionflags=optionflags)
        runner.run(docstring)
        results = runner.summarize()
        if results.failed:
            raise AssertionError(results)
