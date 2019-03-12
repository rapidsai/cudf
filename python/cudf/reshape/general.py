# Copyright (c) 2018, NVIDIA CORPORATION.

import numpy as np
import pandas as pd

from cudf.dataframe import Series
from cudf.dataframe import Buffer
from cudf.dataframe import DataFrame
from cudf.utils import cudautils
from cudf.dataframe.categorical import CategoricalColumn


def melt(frame, id_vars=None, value_vars=None, var_name='variable',
         value_name='value'):
    """Unpivots a DataFrame from wide format to long format,
    optionally leaving identifier variables set.

    Parameters
    ----------
    frame : DataFrame
    id_vars : tuple, list, or ndarray, optional
        Column(s) to use as identifier variables.
        default: None
    value_vars : tuple, list, or ndarray, optional
        Column(s) to unpivot.
        default: all columns that are not set as `id_vars`.
    var_name : scalar
        Name to use for the `variable` column.
        default: frame.columns.name or 'variable'
    value_name : str
        Name to use for the `value` column.
        default: 'value'

    Returns
    -------
    out : DataFrame
        Melted result

    Difference from pandas:
     * Does not support 'col_level' because cuDF does not have multi-index

    Examples
    --------
    >>> import cudf
    >>> import numpy as np
    >>> df = cudf.DataFrame({'A': {0: 1, 1: 1, 2: 5},
    ...                      'B': {0: 1, 1: 3, 2: 6},
    ...                      'C': {0: 1.0, 1: np.nan, 2: 4.0},
    ...                      'D': {0: 2.0, 1: 5.0, 2: 6.0}})
    >>> cudf.melt(frame=df, id_vars=['A', 'B'], value_vars=['C', 'D'])
         A    B variable value
    0    1    1        C   1.0
    1    1    3        C
    2    5    6        C   4.0
    3    1    1        D   2.0
    4    1    3        D   5.0
    5    5    6        D   6.0
    """
    # Arg cleaning
    import collections
    # id_vars
    if id_vars is not None:
        if not isinstance(id_vars, collections.abc.Sequence):
            id_vars = [id_vars]
        id_vars = list(id_vars)
        missing = set(id_vars) - set(frame.columns)
        if not len(missing) == 0:
            raise KeyError(
                "The following 'id_vars' are not present"
                " in the DataFrame: {missing}"
                "".format(missing=list(missing)))
    else:
        id_vars = []

    # value_vars
    if value_vars is not None:
        if not isinstance(value_vars, collections.abc.Sequence):
            value_vars = [value_vars]
        value_vars = list(value_vars)
        missing = set(value_vars) - set(frame.columns)
        if not len(missing) == 0:
            raise KeyError(
                "The following 'value_vars' are not present"
                " in the DataFrame: {missing}"
                "".format(missing=list(missing)))
    else:
        # then all remaining columns in frame
        value_vars = frame.columns.drop(id_vars)
        value_vars = list(value_vars)

    # Error for unimplemented support for datatype
    dtypes = [frame[col].dtype for col in id_vars + value_vars]
    if any(pd.api.types.is_categorical_dtype(t) for t in dtypes):
        raise NotImplementedError('Categorical columns are not yet '
                                  'supported for function')

    # Check dtype homogeneity in value_var
    # Because heterogeneous concat is unimplemented
    dtypes = [frame[col].dtype for col in value_vars]
    if len(dtypes) > 0:
        dtype = dtypes[0]
        if any(t != dtype for t in dtypes):
            raise ValueError('all cols in value_vars must have the same dtype')

    # overlap
    overlap = set(id_vars).intersection(set(value_vars))
    if not len(overlap) == 0:
        raise KeyError(
            "'value_vars' and 'id_vars' cannot have overlap."
            " The following 'value_vars' are ALSO present"
            " in 'id_vars': {overlap}"
            "".format(overlap=list(overlap)))

    N = len(frame)
    K = len(value_vars)

    def _tile(A, reps):
        series_list = [A] * reps
        if reps > 0:
            return Series._concat(objs=series_list, index=None)
        else:
            return Series(Buffer.null(dtype=A.dtype))

    # Step 1: tile id_vars
    mdata = collections.OrderedDict()
    for col in id_vars:
        mdata[col] = _tile(frame[col], K)

    # Step 2: add variable
    var_cols = []
    for i, var in enumerate(value_vars):
        var_cols.append(Series(Buffer(
            cudautils.full(size=N, value=i, dtype=np.int8))))
    temp = Series._concat(objs=var_cols, index=None)
    mdata[var_name] = Series(CategoricalColumn(
        categories=tuple(value_vars), data=temp._column.data, ordered=False))

    # Step 3: add values
    mdata[value_name] = Series._concat(
        objs=[frame[val] for val in value_vars],
        index=None)

    return DataFrame(mdata)
