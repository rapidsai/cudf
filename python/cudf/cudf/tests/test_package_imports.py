
def setup_module(module):
    import cudf
    module.cudfNamespaceVars = cudf.__all__


def test_from_imports():
    before = set(locals().keys())
    from cudf import *
    after = set(locals().keys())
    assert after-before == set(cudfNamespaceVars)
