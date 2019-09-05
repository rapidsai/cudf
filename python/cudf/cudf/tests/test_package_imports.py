import tempfile
import importlib
from os import path
import sys


def setup_module(module):
    """
    Retrieve and save all public symbols in the cudf namespace for comparison
    in tests.
    """
    import cudf
    module.allPossibleCudfNamespaceVars = cudf.__all__
    del cudf
    sys.modules.pop("cudf")


def teardown_function(function):
    """
    "unimport" cudf
    """
    globals().pop("cudf", None)
    sys.modules.pop("cudf", None)


def __getVarsLoaded(module):
    """
    Return the set of vars loaded in the module, minus "extra" vars used by the
    lazy-loading mechanism.
    """
    extraVars = ["import_module", "numba"]
    return set([k for k in module.__dict__.keys()
                if not(k.startswith("__") or (k in extraVars))])


########################################
def test_import_package_module():
    """
    Ensure accessing a namespace var that represents a module in the package
    works. Example: import cudf; cudf.datasets (datasets.py is a mod in the
    cudf package)
    """
    exceptionRaised = None
    import cudf
    try:
        cudf.core
        cudf.datasets
    except Exception as e:
        exceptionRaised = e
    assert exceptionRaised is None


def test_from_import_star():
    """
    Ensure a "from cudf import *" does an "actual" import of everything in the
    cudf namespace.
    """
    # "import *" can only be done at the module level, so create a temp .py
    # file and manually import it.
    fakeModuleFile = tempfile.NamedTemporaryFile(mode="w+", suffix=".py")
    fakeModuleFile.write("from cudf import *\n")
    fakeModuleFile.file.flush()
    spec = importlib.util.spec_from_file_location(
        path.splitext(path.basename(fakeModuleFile.name))[0],
        fakeModuleFile.name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    varsLoaded = __getVarsLoaded(module)
    assert varsLoaded == set(allPossibleCudfNamespaceVars)  # noqa: F821


def test_from_import_names():
    """
    Ensure "from cudf import <name>" works
    """
    from cudf import DataFrame, from_dlpack, arccos  # noqa: F401
    import cudf

    # Only names explicitly imported above should be present
    varsLoaded = __getVarsLoaded(cudf)
    assert varsLoaded == set(["DataFrame", "from_dlpack", "arccos"])


def test_access_names():
    """
    Ensure "import cudf;cudf.<name>" works as expected.
    """
    import cudf
    cudf.Index
    cudf.read_avro
    cudf.arcsin

    # Only names explicitly imported above should be present
    varsLoaded = __getVarsLoaded(cudf)
    assert varsLoaded == set(["Index", "read_avro", "arcsin"])


def test_cuda_context_created():
    """
    Ensure a CUDA context is created even from a lazy import.
    NOTE: This test may need to be run separately and with the --noconftest
    option in order to avoid importing modules that may create a CUDA context
    (via other tests and/or conftest.py)
    """
    import cudf  # noqa: F401
    import numba
    try :
        numba.cuda.current_context()
    # the "cuda" attr of the numba module will not exist if a CUDA context was
    # not created
    except AttributeError:
        assert False, "CUDA context not created"
