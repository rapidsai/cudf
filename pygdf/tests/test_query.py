import inspect

import pytest

from pygdf import queryutils


_params_query_parser = []
_params_query_parser.append(('a > @b', ('a', '__PYGDF_ENVREF__b')))
_params_query_parser.append(('(a + b) <= @c', ('a', 'b', '__PYGDF_ENVREF__c')))
_params_query_parser.append(('a > b if a > 0 else b > a', ('a', 'b')))


@pytest.mark.parametrize('text,expect_args', _params_query_parser)
def test_query_parser(text, expect_args):
    info = queryutils.query_parser(text)
    fn = queryutils.query_builder(info, 'myfoo')
    assert callable(fn)
    argspec = inspect.getargspec(fn)
    assert tuple(argspec.args) == tuple(expect_args)
