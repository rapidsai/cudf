import ast

import six
import numpy as np

from numba import cuda


ENVREF_PREFIX = '__PYGDF_ENVREF__'


class QuerySyntaxError(ValueError):
    pass


class _NameExtractor(ast.NodeVisitor):
    def __init__(self):
        self.colnames = set()
        self.refnames = set()

    def visit_Name(self, node):
        if not isinstance(node.ctx, ast.Load):
            raise QuerySyntaxError('assignment is not allowed')

        name = node.id
        chosen = (self.refnames
                  if name.startswith(ENVREF_PREFIX)
                  else self.colnames)
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
    text = text.replace('@', ENVREF_PREFIX)
    tree = ast.parse(text)
    _check_error(tree)
    [expr] = tree.body
    extractor = _NameExtractor()
    extractor.visit(expr)
    colnames = sorted(extractor.colnames)
    refnames = sorted(extractor.refnames)
    info = {
        'source': text,
        'args': colnames + refnames,
        'colnames': colnames,
        'refnames': refnames,
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
    args = info['args']
    def_line = 'def {funcid}({args}):'.format(funcid=funcid,
                                              args=', '.join(args))
    lines = [def_line, '    return {}'.format(info['source'])]
    source = '\n'.join(lines)
    glbs = {}
    six.exec_(source, glbs)
    return glbs[funcid]


def _check_error(tree):
    if not isinstance(tree, ast.Module):
        raise QuerySyntaxError('top level should be of ast.Module')
    if len(tree.body) != 1:
        raise QuerySyntaxError('too many expressions')


_cache = {}


def query_compile(expr):
    """Compile the query expression.

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

    funcid = 'queryexpr_{:x}'.format(np.uintp(hash(expr)))
    compiled = _cache.get(funcid)
    if compiled is None:
        info = query_parser(expr)
        fn = query_builder(info, funcid)
        args = info['args']
        # compile
        devicefn = cuda.jit(device=True)(fn)

        kernelid = 'kernel_{}'.format(funcid)
        kernel = _wrap_query_expr(kernelid, devicefn, args)

        compiled = info.copy()
        compiled['kernel'] = kernel
        _cache[funcid] = compiled
    return compiled


_kernel_source = '''
@cuda.jit
def {kernelname}(out, {args}):
    idx = cuda.grid(1)
    if idx < out.size:
        out[idx] = queryfn({indiced_args})
'''


def _wrap_query_expr(name, fn, args):
    """Wrap the query expression in a cuda kernel.
    """
    glbls = {
        'queryfn': fn,
        'cuda': cuda,
    }
    kernargs = ['_args_{}'.format(a) for a in args]
    indiced_args = ['{}[idx]'.format(a) for a in kernargs]
    src = _kernel_source.format(kernelname=name,
                                args=', '.join(kernargs),
                                indiced_args=', '.join(indiced_args))
    six.exec_(src, glbls)
    kernel = glbls[name]
    return kernel


def query_execute(df, expr):
    """Compile & execute the query expression
    """
    # compile
    compiled = query_compile(expr)
    kernel = compiled['kernel']
    # prepare args
    if compiled['refnames']:
        raise NotImplementedError('env ref not supported yet')
    colarrays = [df[col].to_gpu_array() for col in compiled['colnames']]
    # allocate output buffer
    nrows = len(df)
    out = cuda.device_array(nrows, dtype=np.bool_)
    # run kernel
    args = [out] + colarrays
    kernel.forall(nrows)(*args)
    return out
