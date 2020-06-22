# Copyright (c) 2018, NVIDIA CORPORATION.

import numpy as np
import pandas as pd

import cudf
from cudf.core import DataFrame, Index, Series
from cudf.core.column import as_column, build_categorical_column
from cudf.utils import cudautils
from cudf.utils.dtypes import is_categorical_dtype, is_list_like

_axis_map = {0: 0, 1: 1, "index": 0, "columns": 1}


def _align_objs(objs, how="outer"):
    """Align a set of Series or Dataframe objects.

    Parameters
    ----------
    objs : list of DataFrame, Series, or Index

    Returns
    -------
    A bool for if indexes have matched and a set of
    reindexed and aligned objects ready for concatenation
    """
    # Check if multiindex then check if indexes match. GenericIndex
    # returns ndarray tuple of bools requiring additional filter.
    # Then check for duplicate index value.
    i_objs = iter(objs)
    first = next(i_objs)
    match_index = all(first.index.equals(rest.index) for rest in i_objs)

    if match_index:
        return objs, True
    else:
        if not all(o.index.is_unique for o in objs):
            raise ValueError("cannot reindex from a duplicate axis")

        index = objs[0].index
        for obj in objs[1:]:
            name = index.name
            index = (
                cudf.DataFrame(index=obj.index)
                .join(cudf.DataFrame(index=index), how=how)
                .index
            )
            index.name = name

        return [obj.reindex(index) for obj in objs], False


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

    objs = [obj for obj in objs if obj is not None]

    # Return for single object
    if len(objs) == 1:
        return objs[0]

    if len(objs) == 0:
        raise ValueError("All objects passed were None")

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

        sr_name = 0
        for idx, o in enumerate(objs):
            if isinstance(o, Series):
                name = o.name
                if name is None:
                    name = sr_name
                    sr_name += 1
                objs[idx] = o.to_frame(name=name)

        objs, match_index = _align_objs(objs)

        for idx, o in enumerate(objs):
            if not ignore_index and idx == 0:
                df.index = o.index
            for col in o._data.names:
                if col in df._data:
                    raise NotImplementedError(
                        "A Column with duplicate name found: {0}, cuDF\
                        doesn't support having multiple columns with\
                        same names yet.".format(
                            col
                        )
                    )
                df[col] = o._data[col]

        result_columns = objs[0].columns
        for o in objs[1:]:
            result_columns = result_columns.append(o.columns)

        df.columns = result_columns.unique()
        if ignore_index:
            df.index = None
            return df
        elif not match_index:
            return df.sort_index()
        else:
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
            categories=value_vars,
            codes=as_column(temp._column.base_data, dtype=temp._column.dtype),
            mask=temp._column.base_mask,
            size=temp._column.size,
            offset=temp._column.offset,
            ordered=False,
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
        result_df = df.drop(labels=columns)
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
            for col in col_enc_df.columns.difference(df._data.names):
                result_df[col] = col_enc_df._data[col]

        return result_df


def merge_sorted(
    objs,
    keys=None,
    by_index=False,
    ignore_index=False,
    ascending=True,
    na_position="last",
):
    """Merge a list of sorted DataFrame or Series objects.

    Dataframes/Series in objs list MUST be pre-sorted by columns
    listed in `keys`, or by the index (if `by_index=True`).

    Parameters
    ----------
    objs : list of DataFrame, Series, or Index
    keys : list, default None
        List of Column names to sort by. If None, all columns used
        (Ignored if `index=True`)
    by_index : bool, default False
        Use index for sorting. `keys` input will be ignored if True
    ignore_index : bool, default False
        Drop and ignore index during merge. Default range index will
        be used in the output dataframe.
    ascending : bool, default True
        Sorting is in ascending order, otherwise it is descending
    na_position : {‘first’, ‘last’}, default ‘last’
        'first' nulls at the beginning, 'last' nulls at the end

    Returns
    -------
    A new, lexicographically sorted, DataFrame/Series.
    """

    if not pd.api.types.is_list_like(objs):
        raise TypeError("objs must be a list-like of Frame-like objects")

    if len(objs) < 1:
        raise ValueError("objs must be non-empty")

    if not all(isinstance(table, cudf.core.frame.Frame) for table in objs):
        raise TypeError("Elements of objs must be Frame-like")

    if len(objs) == 1:
        return objs[0]

    if by_index and ignore_index:
        raise ValueError("`by_index` and `ignore_index` cannot both be True")

    result = objs[0].__class__._from_table(
        cudf._lib.merge.merge_sorted(
            objs,
            keys=keys,
            by_index=by_index,
            ignore_index=ignore_index,
            ascending=ascending,
            na_position=na_position,
        )
    )
    result._copy_categories(objs[0])
    return result
