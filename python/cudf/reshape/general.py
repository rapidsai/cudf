# Copyright (c) 2018, NVIDIA CORPORATION.

import numpy as np

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
    molten : DataFrame

    Difference from pandas:
     * Does not support 'col_level' because cuDF does not have multi-index

    TODO: Examples
    """

    # Arg cleaning
    import types
    # id_vars
    if id_vars is not None:
        if not isinstance(id_vars, list):
            id_vars = [id_vars]
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
        if not isinstance(value_vars, list):
            value_vars = [value_vars]
        missing = set(value_vars) - set(frame.columns)
        if not len(missing) == 0:
            raise KeyError(
                "The following 'value_vars' are not present"
                " in the DataFrame: {missing}"
                "".format(missing=list(missing)))
    else:
        value_vars = []

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
        return Series._concat(objs=series_list, index=None)

    # Step 1: tile id_vars
    mdata = {}
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
