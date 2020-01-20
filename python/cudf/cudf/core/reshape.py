# Copyright (c) 2018, NVIDIA CORPORATION.

import numpy as np

import cudf
from cudf.core import DataFrame, Index, MultiIndex, RangeIndex, Series
from cudf.core.column import build_categorical_column
from cudf.core.index import as_index
from cudf.utils import cudautils
from cudf.utils.dtypes import is_categorical_dtype, is_list_like

_axis_map = {0: 0, 1: 1, "index": 0, "columns": 1}


def concat(objs, axis=0, ignore_index=False, sort=None):
    """Concatenate DataFrames, Series, or Indices row-wise.

    Parameters
    ----------
    objs : list of DataFrame, Series, or Index
    axis : {0/'index', 1/'columns'}, default 0
        The axis to concatenate along.
    ignore_index : bool, default False
        Set True to ignore the index of the *objs* and provide a
        default range index instead.

    Returns
    -------
    A new object of like type with rows from each object in ``objs``.
    """
    if sort not in (None, False):
        raise NotImplementedError("sort parameter is not yet supported")

    if not objs:
        raise ValueError("Need at least one object to concatenate")

    # no-op for single object
    if len(objs) == 1:
        return objs[0]

    # convert any RangeIndex
    objs = [
        as_index(obj._values) if isinstance(obj, RangeIndex) else obj
        for obj in objs
    ]

    typs = set(type(o) for o in objs)
    allowed_typs = {Series, DataFrame}

    param_axis = _axis_map.get(axis, None)
    if param_axis is None:
        raise ValueError(
            '`axis` must be 0 / "index" or 1 / "columns", got: {0}'.format(
                param_axis
            )
        )
    else:
        axis = param_axis

    # when axis is 1 (column) we can concat with Series and Dataframes
    if axis == 1:
        assert typs.issubset(allowed_typs)
        df = DataFrame()
        for idx, o in enumerate(objs):
            if idx == 0:
                df.index = o.index
            if isinstance(o, Series):
                name = o.name
                if o.name is None:
                    # pandas uses 0-offset
                    name = idx - 1
                df[name] = o._column
            else:
                for col in o.columns:
                    df[col] = o[col]._column
        if isinstance(objs[0], DataFrame) and isinstance(
            objs[0].columns, MultiIndex
        ):
            df.columns = MultiIndex._concat([obj.columns for obj in objs])
        return df

    typ = list(typs)[0]

    if len(typs) > 1:
        raise ValueError(
            "`concat` expects all objects to be of the same "
            "type. Got mix of %r." % [t.__name__ for t in typs]
        )
    typ = list(typs)[0]

    if typ is DataFrame:
        return DataFrame._concat(objs, axis=axis, ignore_index=ignore_index)
    elif typ is Series:
        return Series._concat(objs, axis=axis)
    elif typ is cudf.MultiIndex:
        return cudf.MultiIndex._concat(objs)
    elif issubclass(typ, Index):
        return Index._concat(objs)
    else:
        raise ValueError("Unknown type %r" % typ)


def melt(
    frame,
    id_vars=None,
    value_vars=None,
    var_name=None,
    value_name="value",
    col_level=None,
):
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
    assert col_level in (None,)

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
                "".format(missing=list(missing))
            )
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
                "".format(missing=list(missing))
            )
    else:
        # then all remaining columns in frame
        value_vars = frame.columns.drop(id_vars)
        value_vars = list(value_vars)

    # Error for unimplemented support for datatype
    dtypes = [frame[col].dtype for col in id_vars + value_vars]
    if any(is_categorical_dtype(t) for t in dtypes):
        raise NotImplementedError(
            "Categorical columns are not yet " "supported for function"
        )

    # Check dtype homogeneity in value_var
    # Because heterogeneous concat is unimplemented
    dtypes = [frame[col].dtype for col in value_vars]
    if len(dtypes) > 0:
        dtype = dtypes[0]
        if any(t != dtype for t in dtypes):
            raise ValueError("all cols in value_vars must have the same dtype")

    # overlap
    overlap = set(id_vars).intersection(set(value_vars))
    if not len(overlap) == 0:
        raise KeyError(
            "'value_vars' and 'id_vars' cannot have overlap."
            " The following 'value_vars' are ALSO present"
            " in 'id_vars': {overlap}"
            "".format(overlap=list(overlap))
        )

    N = len(frame)
    K = len(value_vars)

    def _tile(A, reps):
        series_list = [A] * reps
        if reps > 0:
            return Series._concat(objs=series_list, index=None)
        else:
            return Series([], dtype=A.dtype)

    # Step 1: tile id_vars
    mdata = collections.OrderedDict()
    for col in id_vars:
        mdata[col] = _tile(frame[col], K)

    # Step 2: add variable
    var_cols = []
    for i, var in enumerate(value_vars):
        var_cols.append(Series(cudautils.full(size=N, value=i, dtype=np.int8)))
    temp = Series._concat(objs=var_cols, index=None)

    if not var_name:
        var_name = "variable"

    mdata[var_name] = Series(
        build_categorical_column(
            categories=value_vars, codes=temp._column, ordered=False
        )
    )

    # Step 3: add values
    mdata[value_name] = Series._concat(
        objs=[frame[val] for val in value_vars], index=None
    )

    return DataFrame(mdata)


def get_dummies(
    df,
    prefix=None,
    prefix_sep="_",
    dummy_na=False,
    columns=None,
    cats={},
    sparse=False,
    drop_first=False,
    dtype="int8",
):
    """ Returns a dataframe whose columns are the one hot encodings of all
    columns in `df`

    Parameters
    ----------
    df : cudf.DataFrame
        dataframe to encode
    prefix : str, dict, or sequence, optional
        prefix to append. Either a str (to apply a constant prefix), dict
        mapping column names to prefixes, or sequence of prefixes to apply with
        the same length as the number of columns. If not supplied, defaults
        to the empty string
    prefix_sep : str, dict, or sequence, optional, default '_'
        separator to use when appending prefixes
    dummy_na : boolean, optional
        Right now this is NON-FUNCTIONAL argument in rapids.
    cats : dict, optional
        dictionary mapping column names to sequences of integers representing
        that column's category. See `cudf.DataFrame.one_hot_encoding` for more
        information. if not supplied, it will be computed
    sparse : boolean, optional
        Right now this is NON-FUNCTIONAL argument in rapids.
    drop_first : boolean, optional
        Right now this is NON-FUNCTIONAL argument in rapids.
    columns : sequence of str, optional
        Names of columns to encode. If not provided, will attempt to encode all
        columns. Note this is different from pandas default behavior, which
        encodes all columns with dtype object or categorical
    dtype : str, optional
        output dtype, default 'int8'
    """
    if dummy_na:
        raise NotImplementedError("dummy_na is not supported yet")

    if sparse:
        raise NotImplementedError("sparse is not supported yet")

    if drop_first:
        raise NotImplementedError("drop_first is not supported yet")

    # TODO: This has to go away once we start supporting uint8.
    if dtype == np.uint8:
        dtype = "int8"

    encode_fallback_dtypes = ["object", "category"]

    if columns is None or len(columns) == 0:
        columns = df.select_dtypes(include=encode_fallback_dtypes).columns

    def length_check(obj, name):
        err_msg = (
            "Length of '{name}' ({len_obj}) did not match the "
            "length of the columns being encoded ({len_required})."
        )

        if is_list_like(obj):
            if len(obj) != len(columns):
                err_msg = err_msg.format(
                    name=name, len_obj=len(obj), len_required=len(columns)
                )
                raise ValueError(err_msg)

    length_check(prefix, "prefix")
    length_check(prefix_sep, "prefix_sep")

    if prefix is None:
        prefix = columns

    if isinstance(prefix, str):
        prefix_map = {}
    elif isinstance(prefix, dict):
        prefix_map = prefix
    else:
        prefix_map = dict(zip(columns, prefix))

    if isinstance(prefix_sep, str):
        prefix_sep_map = {}
    elif isinstance(prefix_sep, dict):
        prefix_sep_map = prefix_sep
    else:
        prefix_sep_map = dict(zip(columns, prefix_sep))

    # If we have no columns to encode, we need to drop fallback columns(if any)
    if len(columns) == 0:
        return df.select_dtypes(exclude=encode_fallback_dtypes)
    else:
        df_list = []

        for name in columns:
            if hasattr(df[name]._column, "categories"):
                unique = df[name]._column.categories
            else:
                unique = df[name].unique()

            col_enc_df = df.one_hot_encoding(
                name,
                prefix=prefix_map.get(name, prefix),
                cats=cats.get(name, unique),
                prefix_sep=prefix_sep_map.get(name, prefix_sep),
                dtype=dtype,
            )
            df_list.append(col_enc_df)

        return concat(df_list, axis=1).drop(labels=columns)
