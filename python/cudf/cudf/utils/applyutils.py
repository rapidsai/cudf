# Copyright (c) 2018-2024, NVIDIA CORPORATION.
from __future__ import annotations

import functools
from typing import Any

import cupy as cp
from numba import cuda
from numba.core.utils import pysignature

import cudf
from cudf.core._internals import binaryop
from cudf.core.buffer import acquire_spill_lock
from cudf.core.column import column
from cudf.utils import utils
from cudf.utils._numba import _CUDFNumbaConfig
from cudf.utils.docutils import docfmt_partial

_doc_applyparams = """
df : DataFrame
    The source dataframe.
func : function
    The transformation function that will be executed on the CUDA GPU.
incols: list or dict
    A list of names of input columns that match the function arguments.
    Or, a dictionary mapping input column names to their corresponding
    function arguments such as {'col1': 'arg1'}.
outcols: dict
    A dictionary of output column names and their dtype.
kwargs: dict
    name-value of extra arguments.  These values are passed
    directly into the function.
pessimistic_nulls : bool
    Whether or not apply_rows output should be null when any corresponding
    input is null. If False, all outputs will be non-null, but will be the
    result of applying func against the underlying column data, which
    may be garbage.
"""

_doc_applychunkparams = """
chunks : int or Series-like
    If it is an ``int``, it is the chunksize.
    If it is an array, it contains integer offset for the start of each chunk.
    The span of a chunk for chunk i-th is ``data[chunks[i] : chunks[i + 1]]``
    for any ``i + 1 < chunks.size``; or, ``data[chunks[i]:]`` for the
    ``i == len(chunks) - 1``.
tpb : int; optional
    The threads-per-block for the underlying kernel.
    If not specified (Default), uses Numba ``.forall(...)`` built-in to query
    the CUDA Driver API to determine optimal kernel launch configuration.
    Specify 1 to emulate serial execution for each chunk.  It is a good
    starting point but inefficient.
    Its maximum possible value is limited by the available CUDA GPU resources.
blkct : int; optional
    The number of blocks for the underlying kernel.
    If not specified (Default) and ``tpb`` is not specified (Default), uses
    Numba ``.forall(...)`` built-in to query the CUDA Driver API to determine
    optimal kernel launch configuration.
    If not specified (Default) and ``tpb`` is specified, uses ``chunks`` as the
    number of blocks.
"""

doc_apply = docfmt_partial(params=_doc_applyparams)
doc_applychunks = docfmt_partial(
    params=_doc_applyparams, params_chunks=_doc_applychunkparams
)


@doc_apply()
def apply_rows(
    df, func, incols, outcols, kwargs, pessimistic_nulls, cache_key
):
    """Row-wise transformation

    Parameters
    ----------
    {params}
    """
    applyrows = ApplyRowsCompiler(
        func, incols, outcols, kwargs, pessimistic_nulls, cache_key=cache_key
    )
    return applyrows.run(df)


@doc_applychunks()
def apply_chunks(
    df,
    func,
    incols,
    outcols,
    kwargs,
    pessimistic_nulls,
    chunks,
    blkct=None,
    tpb=None,
):
    """Chunk-wise transformation

    Parameters
    ----------
    {params}
    {params_chunks}
    """
    applychunks = ApplyChunksCompiler(
        func, incols, outcols, kwargs, pessimistic_nulls, cache_key=None
    )
    return applychunks.run(df, chunks=chunks, tpb=tpb)


@acquire_spill_lock()
def make_aggregate_nullmask(df, columns=None, op="__and__"):
    out_mask = None
    for k in columns or df._data:
        col = cudf.core.dataframe.extract_col(df, k)
        if not col.nullable:
            continue
        nullmask = column.as_column(df[k]._column.nullmask)

        if out_mask is None:
            out_mask = column.as_column(
                nullmask.copy(), dtype=utils.mask_dtype
            )
        else:
            out_mask = binaryop.binaryop(
                nullmask, out_mask, op, out_mask.dtype
            )

    return out_mask


class ApplyKernelCompilerBase:
    def __init__(
        self, func, incols, outcols, kwargs, pessimistic_nulls, cache_key
    ):
        # Get signature of user function
        sig = pysignature(func)
        self.sig = sig
        self.incols = incols
        self.outcols = outcols
        self.kwargs = kwargs
        self.pessimistic_nulls = pessimistic_nulls
        self.cache_key = cache_key
        self.kernel = self.compile(func, sig.parameters.keys(), kwargs.keys())

    @acquire_spill_lock()
    def run(self, df, **launch_params):
        # Get input columns
        if isinstance(self.incols, dict):
            inputs = {
                v: df[k]._column.data_array_view(mode="read")
                for (k, v) in self.incols.items()
            }
        else:
            inputs = {
                k: df[k]._column.data_array_view(mode="read")
                for k in self.incols
            }
        # Allocate output columns
        outputs = {}
        for k, dt in self.outcols.items():
            outputs[k] = column.column_empty(
                len(df), dt, False
            ).data_array_view(mode="write")
        # Bind argument
        args = {}
        for dct in [inputs, outputs, self.kwargs]:
            args.update(dct)
        bound = self.sig.bind(**args)
        # Launch kernel
        self.launch_kernel(df, bound.args, **launch_params)
        # Prepare pessimistic nullmask
        if self.pessimistic_nulls:
            out_mask = make_aggregate_nullmask(df, columns=self.incols)
        else:
            out_mask = None
        # Prepare output frame
        outdf = df.copy()
        for k in sorted(self.outcols):
            outdf[k] = cudf.Series(
                outputs[k], index=outdf.index, nan_as_null=False
            )
            if out_mask is not None:
                outdf._data[k] = outdf[k]._column.set_mask(
                    out_mask.data_array_view(mode="write")
                )

        return outdf


class ApplyRowsCompiler(ApplyKernelCompilerBase):
    def compile(self, func, argnames, extra_argnames):
        # Compile kernel
        kernel = _load_cache_or_make_row_wise_kernel(
            self.cache_key, func, argnames, extra_argnames
        )
        return kernel

    def launch_kernel(self, df, args):
        with _CUDFNumbaConfig():
            self.kernel.forall(len(df))(*args)


class ApplyChunksCompiler(ApplyKernelCompilerBase):
    def compile(self, func, argnames, extra_argnames):
        # Compile kernel
        kernel = _load_cache_or_make_chunk_wise_kernel(
            func, argnames, extra_argnames
        )
        return kernel

    def launch_kernel(self, df, args, chunks, blkct=None, tpb=None):
        chunks = self.normalize_chunks(len(df), chunks)
        if blkct is None and tpb is None:
            with _CUDFNumbaConfig():
                self.kernel.forall(len(df))(len(df), chunks, *args)
        else:
            assert tpb is not None
            if blkct is None:
                blkct = chunks.size
            with _CUDFNumbaConfig():
                self.kernel[blkct, tpb](len(df), chunks, *args)

    def normalize_chunks(self, size, chunks):
        if isinstance(chunks, int):
            # *chunks* is the chunksize
            return cuda.as_cuda_array(
                cp.arange(start=0, stop=size, step=chunks)
            ).view("int64")
        else:
            # *chunks* is an array of chunk leading offset
            return cuda.as_cuda_array(cp.asarray(chunks)).view("int64")


def _make_row_wise_kernel(func, argnames, extras):
    """
    Make a kernel that does a stride loop over the input rows.

    Each thread is responsible for a row in each iteration.
    Several iteration may be needed to handling a large number of rows.

    The resulting kernel can be used with any 1D grid size and 1D block size.
    """
    # Build kernel source
    argnames = list(map(_mangle_user, argnames))
    extras = list(map(_mangle_user, extras))
    source = """
def row_wise_kernel({args}):
{body}
"""

    args = ", ".join(argnames)
    body = []

    body.append("tid = cuda.grid(1)")
    body.append("ntid = cuda.gridsize(1)")

    for a in argnames:
        if a not in extras:
            start = "tid"
            stop = ""
            stride = "ntid"
            srcidx = "{a} = {a}[{start}:{stop}:{stride}]"
            body.append(
                srcidx.format(a=a, start=start, stop=stop, stride=stride)
            )

    body.append(f"inner({args})")

    indented = ["{}{}".format(" " * 4, ln) for ln in body]
    # Finalize source
    concrete = source.format(args=args, body="\n".join(indented))
    # Get bytecode
    glbs = {"inner": cuda.jit(device=True)(func), "cuda": cuda}
    exec(concrete, glbs)
    # Compile as CUDA kernel
    kernel = cuda.jit(glbs["row_wise_kernel"])
    return kernel


def _make_chunk_wise_kernel(func, argnames, extras):
    """
    Make a kernel that does a stride loop over the input chunks.

    Each block is responsible for a chunk in each iteration.
    Several iteration may be needed to handling a large number of chunks.

    The user function *func* will have all threads in the block for its
    computation.

    The resulting kernel can be used with any 1D grid size and 1D block size.
    """

    # Build kernel source
    argnames = list(map(_mangle_user, argnames))
    extras = list(map(_mangle_user, extras))
    source = """
def chunk_wise_kernel(nrows, chunks, {args}):
{body}
"""

    args = ", ".join(argnames)
    body = []

    body.append("blkid = cuda.blockIdx.x")
    body.append("nblkid = cuda.gridDim.x")
    body.append("tid = cuda.threadIdx.x")
    body.append("ntid = cuda.blockDim.x")

    # Stride loop over the block
    body.append("for curblk in range(blkid, chunks.size, nblkid):")
    indent = " " * 4

    body.append(indent + "start = chunks[curblk]")
    body.append(
        indent
        + "stop = chunks[curblk + 1]"
        + " if curblk + 1 < chunks.size else nrows"
    )

    slicedargs = {}
    for a in argnames:
        if a not in extras:
            slicedargs[a] = f"{a}[start:stop]"
        else:
            slicedargs[a] = str(a)
    body.append(
        "{}inner({})".format(
            indent, ", ".join(slicedargs[k] for k in argnames)
        )
    )

    indented = ["{}{}".format(" " * 4, ln) for ln in body]
    # Finalize source
    concrete = source.format(args=args, body="\n".join(indented))
    # Get bytecode
    glbs = {"inner": cuda.jit(device=True)(func), "cuda": cuda}
    exec(concrete, glbs)
    # Compile as CUDA kernel
    kernel = cuda.jit(glbs["chunk_wise_kernel"])
    return kernel


_cache: dict[Any, Any] = dict()


@functools.wraps(_make_row_wise_kernel)
def _load_cache_or_make_row_wise_kernel(cache_key, func, *args, **kwargs):
    """Caching version of ``_make_row_wise_kernel``."""
    if cache_key is None:
        cache_key = func
    try:
        out = _cache[cache_key]
        # print("apply cache loaded", cache_key)
        return out
    except KeyError:
        # print("apply cache NOT loaded", cache_key)
        kernel = _make_row_wise_kernel(func, *args, **kwargs)
        _cache[cache_key] = kernel
        return kernel


@functools.wraps(_make_chunk_wise_kernel)
def _load_cache_or_make_chunk_wise_kernel(func, *args, **kwargs):
    """Caching version of ``_make_row_wise_kernel``."""
    try:
        return _cache[func]
    except KeyError:
        kernel = _make_chunk_wise_kernel(func, *args, **kwargs)
        _cache[func] = kernel
        return kernel


def _mangle_user(name):
    """Mangle user variable name"""
    return f"__user_{name}"
