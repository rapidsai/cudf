# Copyright (c) 2018-2024, NVIDIA CORPORATION.
from __future__ import annotations

import ast
import datetime
from typing import Any

import numpy as np
from numba import cuda

import cudf
from cudf.core.buffer import acquire_spill_lock
from cudf.core.column import column_empty
from cudf.utils import applyutils
from cudf.utils._numba import _CUDFNumbaConfig
from cudf.utils.dtypes import (
    BOOL_TYPES,
    DATETIME_TYPES,
    NUMERIC_TYPES,
    TIMEDELTA_TYPES,
)

ENVREF_PREFIX = "__CUDF_ENVREF__"

SUPPORTED_QUERY_TYPES = {
    np.dtype(dt)
    for dt in NUMERIC_TYPES | DATETIME_TYPES | TIMEDELTA_TYPES | BOOL_TYPES
}


class QuerySyntaxError(ValueError):
    pass


class _NameExtractor(ast.NodeVisitor):
    def __init__(self):
        self.colnames = set()
        self.refnames = set()

    def visit_Name(self, node):
        if not isinstance(node.ctx, ast.Load):
            raise QuerySyntaxError("assignment is not allowed")

        name = node.id
        chosen = (
            self.refnames if name.startswith(ENVREF_PREFIX) else self.colnames
        )
        chosen.add(name)


def query_parser(text):
    """The query expression parser.

    See https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.query.html

    * names with '@' prefix are global reference.
    * other names must be column names of the dataframe.

    Parameters
    ----------
    text: str
        The query string

    Returns
    -------
    info: a `dict` of the parsed info
    """
    # convert any '@' to
    text = text.replace("@", ENVREF_PREFIX)
    tree = ast.parse(text)
    _check_error(tree)
    [expr] = tree.body
    extractor = _NameExtractor()
    extractor.visit(expr)
    colnames = sorted(extractor.colnames)
    refnames = sorted(extractor.refnames)
    info = {
        "source": text,
        "args": colnames + refnames,
        "colnames": colnames,
        "refnames": refnames,
    }
    return info


def query_builder(info, funcid):
    """Function builder for the query expression

    Parameters
    ----------
    info: dict
        From the `query_parser()`
    funcid: str
        The name for the function being generated

    Returns
    -------
    func: a python function of the query
    """
    args = info["args"]
    def_line = "def {funcid}({args}):".format(
        funcid=funcid, args=", ".join(args)
    )
    lines = [def_line, "    return {}".format(info["source"])]
    source = "\n".join(lines)
    glbs = {}
    exec(source, glbs)
    return glbs[funcid]


def _check_error(tree):
    if not isinstance(tree, ast.Module):
        raise QuerySyntaxError("top level should be of ast.Module")
    if len(tree.body) != 1:
        raise QuerySyntaxError("too many expressions")


_cache: dict[Any, Any] = {}


def query_compile(expr):
    """Compile the query expression.

    This generates a CUDA Kernel for the query expression.  The kernel is
    cached for reuse.  All variable names, including both references to
    columns and references to variables in the calling environment, in the
    expression are passed as argument to the kernel. Thus, the kernel is
    reusable on any dataframe and in any environment.

    Parameters
    ----------
    expr : str
        The boolean expression

    Returns
    -------
    compiled: dict
        key "kernel" is the cuda kernel for the query.
        key "args" is a sequence of name of the arguments.
    """

    # hash returns in the semi-open interval [-2**63, 2**63)
    funcid = f"queryexpr_{(hash(expr) + 2**63):x}"
    # Load cache
    compiled = _cache.get(funcid)
    # Cache not found
    if compiled is None:
        info = query_parser(expr)
        fn = query_builder(info, funcid)
        args = info["args"]
        # compile
        devicefn = cuda.jit(device=True)(fn)

        kernelid = f"kernel_{funcid}"
        kernel = _wrap_query_expr(kernelid, devicefn, args)

        compiled = info.copy()
        compiled["kernel"] = kernel
        # Store cache
        _cache[funcid] = compiled
    return compiled


_kernel_source = """
@cuda.jit
def {kernelname}(out, {args}):
    idx = cuda.grid(1)
    if idx < out.size:
        out[idx] = queryfn({indiced_args})
"""


def _wrap_query_expr(name, fn, args):
    """Wrap the query expression in a cuda kernel."""

    def _add_idx(arg):
        if arg.startswith(ENVREF_PREFIX):
            return arg
        else:
            return f"{arg}[idx]"

    def _add_prefix(arg):
        return f"_args_{arg}"

    glbls = {"queryfn": fn, "cuda": cuda}
    kernargs = map(_add_prefix, args)
    indiced_args = map(_add_prefix, map(_add_idx, args))
    src = _kernel_source.format(
        kernelname=name,
        args=", ".join(kernargs),
        indiced_args=", ".join(indiced_args),
    )
    exec(src, glbls)
    kernel = glbls[name]
    return kernel


@acquire_spill_lock()
def query_execute(df, expr, callenv):
    """Compile & execute the query expression

    Note: the expression is compiled and cached for future reuse.

    Parameters
    ----------
    df : DataFrame
    expr : str
        boolean expression
    callenv : dict
        Contains keys 'local_dict', 'locals' and 'globals' which are all dict.
        They represent the arg, local and global dictionaries of the caller.
    """
    # compile
    compiled = query_compile(expr)
    columns = compiled["colnames"]

    # prepare col args
    colarrays = [cudf.core.dataframe.extract_col(df, col) for col in columns]

    # wait to check the types until we know which cols are used
    if any(col.dtype not in SUPPORTED_QUERY_TYPES for col in colarrays):
        raise TypeError(
            "query only supports numeric, datetime, timedelta, "
            "or bool dtypes."
        )

    colarrays = [col.data_array_view(mode="read") for col in colarrays]

    kernel = compiled["kernel"]
    # process env args
    envargs = []
    envdict = callenv["globals"].copy()
    envdict.update(callenv["locals"])
    envdict.update(callenv["local_dict"])
    for name in compiled["refnames"]:
        name = name[len(ENVREF_PREFIX) :]
        try:
            val = envdict[name]
            if isinstance(val, datetime.datetime):
                val = np.datetime64(val)
        except KeyError:
            msg = "{!r} not defined in the calling environment"
            raise NameError(msg.format(name))
        else:
            envargs.append(val)

    # allocate output buffer
    nrows = len(df)
    out = column_empty(nrows, dtype=np.bool_, for_numba=True)
    # run kernel
    args = [out, *colarrays, *envargs]
    with _CUDFNumbaConfig():
        kernel.forall(nrows)(*args)
    out_mask = applyutils.make_aggregate_nullmask(df, columns=columns)
    return out.set_mask(out_mask).fillna(False)
