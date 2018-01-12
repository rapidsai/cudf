from weakref import WeakKeyDictionary
import functools

from numba.utils import pysignature, exec_
from numba import cuda


def apply_rows(df, func, incols, outcols, kwargs):
    # Get input columns
    inputs = {k: df[k].to_gpu_array() for k in incols}
    # Allocate output columns
    outputs = {}
    for k, dt in outcols.items():
        outputs[k] = cuda.device_array(len(df), dtype=dt)
    # Get signature of user function
    sig = pysignature(func)
    # Compile kernel
    kernel = _load_cache_or_make_row_wise_kernel(func, sig.parameters.keys(),
                                                 kwargs.keys())
    # Bind argument
    args = {}
    for dct in [inputs, outputs, kwargs]:
        args.update(dct)
    bound = sig.bind(**args)
    # Launch kernel
    blksz = 64
    blkct = min(16, max(1, len(df) // blksz))
    kernel[blkct, blksz](*bound.args)
    # Prepare output frame
    outdf = df.copy()
    for k in sorted(outcols):
        outdf[k] = outputs[k]
    return outdf


def _make_row_wise_kernel(func, argnames, extras):
    """
    Make a kernel that does a stride loop over the input columns.
    """
    # Build kernel source
    argnames = list(map(_mangle_user, argnames))
    extras = list(map(_mangle_user, extras))
    source = """
def elemwise({args}):
{body}
"""

    args = ', '.join(argnames)
    body = []

    body.append('tid = cuda.grid(1)')
    body.append('ntid = cuda.gridsize(1)')

    for a in argnames:
        if a not in extras:
            start = 'tid'
            stop = ''
            stride = 'ntid'
            srcidx = '{a} = {a}[{start}:{stop}:{stride}]'
            body.append(srcidx.format(a=a, start=start, stop=stop,
                                      stride=stride))

    body.append("inner({})".format(args))

    indented = ['{}{}'.format(' ' * 4, ln) for ln in body]
    # Finalize source
    concrete = source.format(args=args, body='\n'.join(indented))
    # Get bytecode
    glbs = {'inner': cuda.jit(device=True)(func),
            'cuda': cuda}
    exec_(concrete, glbs)
    # Compile as CUDA kernel
    kernel = cuda.jit(glbs['elemwise'])
    return kernel


_cache = WeakKeyDictionary()


@functools.wraps(_make_row_wise_kernel)
def _load_cache_or_make_row_wise_kernel(func, *args, **kwargs):
    """Caching version of ``_make_row_wise_kernel``.
    """
    try:
        return _cache[func]
    except KeyError:
        kernel = _make_row_wise_kernel(func, *args, **kwargs)
        _cache[func] = kernel
        return kernel


def _mangle_user(name):
    """Mangle user variable name
    """
    return "__user_{}".format(name)
