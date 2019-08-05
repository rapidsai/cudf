# Copyright (c) 2018, NVIDIA CORPORATION.

import functools

from numba import cuda, six
from numba.utils import exec_, pysignature

from librmm_cffi import librmm as rmm

import cudf.bindings.binops as cpp_binops
from cudf.dataframe import columnops
from cudf.dataframe.series import Series
from cudf.utils import cudautils, utils
from cudf.utils.docutils import docfmt_partial

_doc_applyparams = """
func : function
    The transformation function that will be executed on the CUDA GPU.
incols: list
    A list of names of input columns.
outcols: dict
    A dictionary of output column names and their dtype.
kwargs: dict
    name-value of extra arguments.  These values are passed
    directly into the function.
"""

_doc_applychunkparams = """
chunks : int or Series-like
            If it is an ``int``, it is the chunksize.
            If it is an array, it contains integer offset for the start of
            each chunk.  The span of a chunk for chunk i-th is
            ``data[chunks[i] : chunks[i + 1]]`` for any
            ``i + 1 < chunks.size``; or, ``data[chunks[i]:]`` for the
            ``i == len(chunks) - 1``.
tpb : int; optional
    It is the thread-per-block for the underlying kernel.
    The default uses 1 thread to emulate serial execution for
    each chunk.  It is a good starting point but inefficient.
    Its maximum possible value is limited by the available CUDA GPU
    resources.
"""

doc_apply = docfmt_partial(params=_doc_applyparams)
doc_applychunks = docfmt_partial(
    params=_doc_applyparams, params_chunks=_doc_applychunkparams
)


@doc_apply()
def apply_rows(df, func, incols, outcols, kwargs, cache_key):
    """Row-wise transformation

    Parameters
    ----------
    df : DataFrame
        The source dataframe.
    {params}

    """
    applyrows = ApplyRowsCompiler(
        func, incols, outcols, kwargs, cache_key=cache_key
    )
    return applyrows.run(df)


@doc_applychunks()
def apply_chunks(df, func, incols, outcols, kwargs, chunks, tpb):
    """Chunk-wise transformation

    Parameters
    ----------
    df : DataFrame
        The source dataframe.
    {params}
    {params_chunks}
    """
    applyrows = ApplyChunksCompiler(
        func, incols, outcols, kwargs, cache_key=None
    )
    return applyrows.run(df, chunks=chunks, tpb=tpb)


def make_aggregate_nullmask(df, columns=None, op="and"):
    out_mask = None
    for k in columns or df.columns:
        if not df[k].has_null_mask:
            continue

        nullmask = df[k].nullmask
        if out_mask is None:
            out_mask = columnops.as_column(
                nullmask.copy(), dtype=utils.mask_dtype
            )
            continue

        cpp_binops.apply_op(
            columnops.as_column(nullmask), out_mask, out_mask, op
        )

    return out_mask


class ApplyKernelCompilerBase(object):
    def __init__(self, func, incols, outcols, kwargs, cache_key):
        # Get signature of user function
        sig = pysignature(func)
        self.sig = sig
        self.incols = incols
        self.outcols = outcols
        self.kwargs = kwargs
        self.cache_key = cache_key
        self.kernel = self.compile(func, sig.parameters.keys(), kwargs.keys())

    def run(self, df, **launch_params):
        # Get input columns
        inputs = {k: df[k].data.mem for k in self.incols}
        # Allocate output columns
        outputs = {}
        for k, dt in self.outcols.items():
            outputs[k] = rmm.device_array(len(df), dtype=dt)
        # Bind argument
        args = {}
        for dct in [inputs, outputs, self.kwargs]:
            args.update(dct)
        bound = self.sig.bind(**args)
        # Launch kernel
        self.launch_kernel(df, bound.args, **launch_params)
        # Prepare pessimistic nullmask
        out_mask = make_aggregate_nullmask(df)
        # Prepare output frame
        outdf = df.copy()
        for k in sorted(self.outcols):
            outdf[k] = outputs[k]
            if out_mask is not None:
                outdf[k] = outdf[k].set_mask(out_mask.data)

        return outdf


class ApplyRowsCompiler(ApplyKernelCompilerBase):
    def compile(self, func, argnames, extra_argnames):
        # Compile kernel
        kernel = _load_cache_or_make_row_wise_kernel(
            self.cache_key, func, argnames, extra_argnames
        )
        return kernel

    def launch_kernel(self, df, args):
        blksz = 64
        blkct = cudautils.optimal_block_count(len(df) // blksz)
        self.kernel[blkct, blksz](*args)


class ApplyChunksCompiler(ApplyKernelCompilerBase):
    def compile(self, func, argnames, extra_argnames):
        # Compile kernel
        kernel = _load_cache_or_make_chunk_wise_kernel(
            func, argnames, extra_argnames
        )
        return kernel

    def launch_kernel(self, df, args, chunks, tpb):
        chunks = self.normalize_chunks(len(df), chunks)
        blkct = cudautils.optimal_block_count(chunks.size)
        self.kernel[blkct, tpb](len(df), chunks, *args)

    def normalize_chunks(self, size, chunks):
        if isinstance(chunks, six.integer_types):
            # *chunks* is the chunksize
            return cudautils.arange(0, size, chunks)
        else:
            # *chunks* is an array of chunk leading offset
            chunks = Series(chunks)
            return chunks.to_gpu_array()


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

    body.append("inner({})".format(args))

    indented = ["{}{}".format(" " * 4, ln) for ln in body]
    # Finalize source
    concrete = source.format(args=args, body="\n".join(indented))
    # Get bytecode
    glbs = {"inner": cuda.jit(device=True)(func), "cuda": cuda}
    exec_(concrete, glbs)
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
            slicedargs[a] = "{}[start:stop]".format(a)
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
    exec_(concrete, glbs)
    # Compile as CUDA kernel
    kernel = cuda.jit(glbs["chunk_wise_kernel"])
    return kernel


_cache = dict()  # WeakKeyDictionary()


@functools.wraps(_make_row_wise_kernel)
def _load_cache_or_make_row_wise_kernel(cache_key, func, *args, **kwargs):
    """Caching version of ``_make_row_wise_kernel``.
    """
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
    """Caching version of ``_make_row_wise_kernel``.
    """
    try:
        return _cache[func]
    except KeyError:
        kernel = _make_chunk_wise_kernel(func, *args, **kwargs)
        _cache[func] = kernel
        return kernel


def _mangle_user(name):
    """Mangle user variable name
    """
    return "__user_{}".format(name)
