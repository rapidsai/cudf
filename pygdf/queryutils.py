import ast

import six


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
    info = {
        'source': text,
        'colnames': extractor.colnames,
        'refnames': extractor.refnames,
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
    args = sorted(info['colnames']) + sorted(info['refnames'])
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

