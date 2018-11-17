from .dataframe import DataFrame
from .series import Series
from .index import Index


def concat(objs, ignore_index=False):
    """Concatenate DataFrames, Series, or Indices row-wise.

    Parameters
    ----------
    objs : list of DataFrame, Series, or Index
    ignore_index : bool
        Set True to ignore the index of the *objs* and provide a
        default range index instead.

    Returns
    -------
    A new object of like type with rows from each object in ``objs``.
    """
    if not objs:
        raise ValueError("Need at least one object to concatenate")

    # no-op for single object
    if len(objs) == 1:
        return objs[0]

    typs = set(type(o) for o in objs)
    if len(typs) > 1:
        raise ValueError("`concat` expects all objects to be of the same "
                         "type. Got mix of %r." % [t.__name__ for t in typs])
    typ = list(typs)[0]

    if typ is DataFrame:
        return DataFrame._concat(objs, ignore_index=ignore_index)
    elif typ is Series:
        return Series._concat(objs)
    elif issubclass(typ, Index):
        return Index._concat(objs)
    else:
        raise ValueError("Unknown type %r" % typ)
