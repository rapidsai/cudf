##################################
# Copyright (c) 2018-2019, NVIDIA CORPORATION.

from importlib import import_module

# Define a mapping between cudf namespace variables and the corresponding
# package/module they're imported from. Actual imports of names into this
# namespace occur when the names are accessed (eg. "import cudf; cudf.<name>")
# or by explicitely importing them (eg. "from cudf import <name>"). Calls to
# "import cudf" will not incur an import cost for variables that are never used.
#
# The map definition is based on the following examples:
#
#    "datasets": (None,)
#  is treated as:
#    from cudf import datasets
#
#    "DataFrame": ("cudf.core",)
#  is treated as:
#    from cudf.core import DataFrame
#
#    "rmm": ("librmm_cffi", "librmm")
#  is treated as:
#    from librmm_cffi import librmm as rmm
#
__namespaceVarModMap = {
    "rmm": ("librmm_cffi", "librmm"),

    "core": (None,),
    "datasets": (None,),

    "DataFrame": ("cudf.core",),
    "Index": ("cudf.core",),
    "MultiIndex": ("cudf.core",),
    "Series": ("cudf.core",),
    "from_pandas": ("cudf.core",),
    "merge": ("cudf.core",),

    "arccos": ("cudf.core.ops",),
    "arcsin": ("cudf.core.ops",),
    "arctan": ("cudf.core.ops",),
    "cos": ("cudf.core.ops",),
    "exp": ("cudf.core.ops",),
    "log": ("cudf.core.ops",),
    "logical_and": ("cudf.core.ops",),
    "logical_not": ("cudf.core.ops",),
    "logical_or": ("cudf.core.ops",),
    "sin": ("cudf.core.ops",),
    "sqrt": ("cudf.core.ops",),
    "tan": ("cudf.core.ops",),

    "concat": ("cudf.core.reshape",),
    "get_dummies": ("cudf.core.reshape",),
    "melt": ("cudf.core.reshape",),

    "from_dlpack": ("cudf.io",),
    "read_avro": ("cudf.io",),
    "read_csv": ("cudf.io",),
    "read_feather": ("cudf.io",),
    "read_hdf": ("cudf.io",),
    "read_json": ("cudf.io",),
    "read_orc": ("cudf.io",),
    "read_parquet": ("cudf.io",),
}

# Note: dunder vars such as __version__ are not part of __all__
__all__ = list(__namespaceVarModMap.keys())
__locals = locals()

# Unconditionally create a CUDA context at import-time
import numba.cuda
numba.cuda.current_context()

def __getattr__(var):
    """
    Return the value of var, updating the the local namespace as a side effect.
    This is only called if 'var' is not defined in the local namespace.
    """
    varInfo = __namespaceVarModMap.get(var)
    val = None

    if varInfo:
        if varInfo[0] is None:
            val = import_module(".%s" % var, package="cudf")
        else:
            mod = import_module(varInfo[0])
            if len(varInfo) == 2:
                val = getattr(mod, varInfo[1])
            else:
                val = getattr(mod, var)

    # special case: compute __version__
    elif var == "__version__":
        from cudf._version import get_versions
        val = get_versions()["version"]

    else:
        raise AttributeError("module: 'cudf' has no attribute '%s'" % var)

    # Save the value in the namespace under the var name.
    # __getattr__() will not be called for this var again.
    __locals[var] = val

    return val
