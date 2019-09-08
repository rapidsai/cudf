import importlib
import sys
import tempfile
from os import path


def setup_module(module):
    """
    Save the original set of imported modules for restoring later and retrieve
    and all public symbols in the cudf namespace for comparison in tests.
    """
    module.origSysMods = dict(sys.modules)
    import cudf

    module.allPossibleCudfNamespaceVars = cudf.__all__
    __unimportCudf()


def teardown_function(function):
    __unimportCudf()


def teardown_module(module):
    """
    Since pytest may have imported cudf modules during the collection phase,
    restore the imports in sys.modules present prior to running these tests, to
    ensure any references to cudf modules created during collection will work.
    """
    sys.modules.update(module.origSysMods)


########################################
def __getVarsLoaded(module):
    """
    Return the set of vars loaded in the module, minus "extra" vars used by the
    lazy-loading mechanism.
    """
    extraVars = ["import_module", "numba", "sys", "CudfModule", "types"]
    return set(
        [
            k
            for k in module.__dict__.keys()
            if not (k.startswith("__") or (k in extraVars))
        ]
    )


def __unimportCudf():
    allCudfSubMods = [m for m in sys.modules.keys() if m.startswith("cudf.")]
    for m in ["cudf"] + allCudfSubMods:
        sys.modules.pop(m, None)


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
        fakeModuleFile.name,
    )
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

    assert DataFrame
    assert from_dlpack
    assert arccos

    varsLoaded = __getVarsLoaded(cudf)
    assert set(["DataFrame", "from_dlpack", "arccos"]).issubset(varsLoaded)
    assert varsLoaded != set(allPossibleCudfNamespaceVars)  # noqa: F821


def test_access_names():
    """
    Ensure "import cudf;cudf.<name>" works as expected.
    """
    import cudf

    assert cudf.Index
    assert cudf.read_avro
    assert cudf.arcsin

    varsLoaded = __getVarsLoaded(cudf)
    assert set(["Index", "read_avro", "arcsin"]).issubset(varsLoaded)
    assert varsLoaded != set(allPossibleCudfNamespaceVars)  # noqa: F821


def test_bad_name():
    """
    Ensure an invalid namespace name raises an exception.
    """
    exceptionRaised = False
    try:
        from cudf import fffdddsssaaa  # noqa: F401
    except Exception:
        # FIXME: do not use a catch-all handler
        exceptionRaised = True

    assert exceptionRaised


def test_version():
    """
    Ensure __version__ is computed and returned
    """
    import cudf

    assert cudf.__version__


def test_cuda_context_created():
    """
    Ensure a CUDA context is created even from a lazy import.
    NOTE: This test may need to be run separately and with the --noconftest
    option in order to avoid importing modules that may create a CUDA context
    (via other tests and/or conftest.py)
    """
    import cudf  # noqa: F401
    import numba

    try:
        numba.cuda.current_context()
    # the "cuda" attr of the numba module will not exist if a CUDA context was
    # not created
    except AttributeError:
        assert False, "CUDA context not created"
