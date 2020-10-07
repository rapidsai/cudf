# Copyright (c) 2018-2020, NVIDIA CORPORATION.
import itertools

import numpy as np
import pandas as pd

import cudf

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


def _normalize_series_and_dataframe(objs, axis):
    sr_name = 0
    for idx, o in enumerate(objs):
        if isinstance(o, cudf.Series):
            if axis == 1:
                name = o.name
                if name is None:
                    name = sr_name
                    sr_name += 1
            else:
                name = sr_name

            objs[idx] = o.to_frame(name=name)


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
    sort : bool, default False
        Sort non-concatenation axis if it is not already aligned.

    Returns
    -------
    A new object of like type with rows from each object in ``objs``.

    Examples
    --------
    Combine two ``Series``.

    >>> import cudf
    >>> s1 = cudf.Series(['a', 'b'])
    >>> s2 = cudf.Series(['c', 'd'])
    >>> s1
    0    a
    1    b
    dtype: object
    >>> s2
    0    c
    1    d
    dtype: object
    >>> cudf.concat([s1, s2])
    0    a
    1    b
    0    c
    1    d
    dtype: object

    Clear the existing index and reset it in the
    result by setting the ``ignore_index`` option to ``True``.

    >>> cudf.concat([s1, s2], ignore_index=True)
    0    a
    1    b
    2    c
    3    d
    dtype: object

    Combine two DataFrame objects with identical columns.

    >>> df1 = cudf.DataFrame([['a', 1], ['b', 2]],
    ...                    columns=['letter', 'number'])
    >>> df1
      letter  number
    0      a       1
    1      b       2
    >>> df2 = cudf.DataFrame([['c', 3], ['d', 4]],
    ...                    columns=['letter', 'number'])
    >>> df2
      letter  number
    0      c       3
    1      d       4
    >>> cudf.concat([df1, df2])
      letter  number
    0      a       1
    1      b       2
    0      c       3
    1      d       4

    Combine DataFrame objects with overlapping columns and return
    everything. Columns outside the intersection will
    be filled with ``null`` values.

    >>> df3 = cudf.DataFrame([['c', 3, 'cat'], ['d', 4, 'dog']],
    ...                    columns=['letter', 'number', 'animal'])
    >>> df3
    letter  number animal
    0      c       3    cat
    1      d       4    dog
    >>> cudf.concat([df1, df3], sort=False)
      letter  number animal
    0      a       1   None
    1      b       2   None
    0      c       3    cat
    1      d       4    dog

    Combine ``DataFrame`` objects horizontally along the
    x axis by passing in ``axis=1``.

    >>> df4 = cudf.DataFrame([['bird', 'polly'], ['monkey', 'george']],
    ...                    columns=['animal', 'name'])
    >>> df4
       animal    name
    0    bird   polly
    1  monkey  george
    >>> cudf.concat([df1, df4], axis=1)
      letter  number  animal    name
    0      a       1    bird   polly
    1      b       2  monkey  george
    """

    if not objs:
        raise ValueError("No objects to concatenate")

    objs = [obj for obj in objs if obj is not None]

    # Return for single object
    if len(objs) == 1:
        if ignore_index:
            result = cudf.DataFrame(
                data=objs[0]._data.copy(deep=True),
                index=cudf.RangeIndex(len(objs[0])),
            )
        else:
            result = objs[0].copy()
        return result

    if len(objs) == 0:
        raise ValueError("All objects passed were None")

    # Retrieve the base types of `objs`. In order to support sub-types
    # and object wrappers, we use `isinstance()` instead of comparing
    # types directly
    typs = set()
    for o in objs:
        if isinstance(o, cudf.MultiIndex):
            typs.add(cudf.MultiIndex)
        if issubclass(type(o), cudf.Index):
            typs.add(type(o))
        elif isinstance(o, cudf.DataFrame):
            typs.add(cudf.DataFrame)
        elif isinstance(o, cudf.Series):
            typs.add(cudf.Series)
        else:
            raise ValueError(f"cannot concatenate object of type {type(o)}")

    allowed_typs = {cudf.Series, cudf.DataFrame}

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
        df = cudf.DataFrame()
        _normalize_series_and_dataframe(objs, axis=axis)

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
            df.index = cudf.RangeIndex(len(objs[0]))
            return df
        elif not match_index:
            return df.sort_index()
        else:
            return df

    typ = list(typs)[0]

    if len(typs) > 1:
        if allowed_typs == typs:
            # This block of code will run when `objs` has
            # both Series & DataFrame kind of inputs.
            _normalize_series_and_dataframe(objs, axis=axis)
            typ = cudf.DataFrame
        else:
            raise ValueError(
                "`concat` cannot concatenate objects of "
                "types: %r." % sorted([t.__name__ for t in typs])
            )

    if typ is cudf.DataFrame:
        objs = [obj for obj in objs if obj.shape != (0, 0)]
        if len(objs) == 0:
            # If objs is empty, that indicates all of
            # objs are empty dataframes.
            return cudf.DataFrame()
        elif len(objs) == 1:
            if ignore_index:
                result = cudf.DataFrame(
                    data=objs[0]._data.copy(deep=True),
                    index=cudf.RangeIndex(len(objs[0])),
                )
            else:
                result = objs[0].copy()
            return result
        else:
            return cudf.DataFrame._concat(
                objs, axis=axis, ignore_index=ignore_index, sort=sort
            )
    elif typ is cudf.Series:
        objs = [obj for obj in objs if len(obj)]
        if len(objs) == 0:
            return cudf.Series()
        elif len(objs) == 1 and not ignore_index:
            return objs[0]
        else:
            return cudf.Series._concat(
                objs, axis=axis, index=None if ignore_index else True
            )
    elif typ is cudf.MultiIndex:
        return cudf.MultiIndex._concat(objs)
    elif issubclass(typ, cudf.Index):
        return cudf.Index._concat(objs)
    else:
        raise ValueError(f"cannot concatenate object of type {typ}")


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
    >>> df = cudf.DataFrame({'A': ['a', 'b', 'c'],
    ...                      'B': [1, 3, 5],
    ...                      'C': [2, 4, 6]})
    >>> df
       A  B  C
    0  a  1  2
    1  b  3  4
    2  c  5  6
    >>> cudf.melt(df, id_vars=['A'], value_vars=['B'])
       A variable  value
    0  a        B      1
    1  b        B      3
    2  c        B      5
    >>> cudf.melt(df, id_vars=['A'], value_vars=['B', 'C'])
       A variable  value
    0  a        B      1
    1  b        B      3
    2  c        B      5
    3  a        C      2
    4  b        C      4
    5  c        C      6

    The names of ‘variable’ and ‘value’ columns can be customized:

    >>> cudf.melt(df, id_vars=['A'], value_vars=['B'],
    ...         var_name='myVarname', value_name='myValname')
       A myVarname  myValname
    0  a         B          1
    1  b         B          3
    2  c         B          5
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
    if any(cudf.utils.dtypes.is_categorical_dtype(t) for t in dtypes):
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
            return cudf.Series._concat(objs=series_list, index=None)
        else:
            return cudf.Series([], dtype=A.dtype)

    # Step 1: tile id_vars
    mdata = collections.OrderedDict()
    for col in id_vars:
        mdata[col] = _tile(frame[col], K)

    # Step 2: add variable
    var_cols = []
    for i, _ in enumerate(value_vars):
        var_cols.append(
            cudf.Series(cudf.core.column.full(N, i, dtype=np.int8))
        )
    temp = cudf.Series._concat(objs=var_cols, index=None)

    if not var_name:
        var_name = "variable"

    mdata[var_name] = cudf.Series(
        cudf.core.column.build_categorical_column(
            categories=value_vars,
            codes=cudf.core.column.as_column(
                temp._column.base_data, dtype=temp._column.dtype
            ),
            mask=temp._column.base_mask,
            size=temp._column.size,
            offset=temp._column.offset,
            ordered=False,
        )
    )

    # Step 3: add values
    mdata[value_name] = cudf.Series._concat(
        objs=[frame[val] for val in value_vars], index=None
    )

    return cudf.DataFrame(mdata)


def get_dummies(
    df,
    prefix=None,
    prefix_sep="_",
    dummy_na=False,
    columns=None,
    cats=None,
    sparse=False,
    drop_first=False,
    dtype="uint8",
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
        Add a column to indicate Nones, if False Nones are ignored.
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
        output dtype, default 'uint8'

    Examples
    --------
    >>> import cudf
    >>> df = cudf.DataFrame({"a": ["value1", "value2", None], "b": [0, 0, 0]})
    >>> cudf.get_dummies(df)
       b  a_value1  a_value2
    0  0         1         0
    1  0         0         1
    2  0         0         0

    >>> cudf.get_dummies(df, dummy_na=True)
       b  a_None  a_value1  a_value2
    0  0       0         1         0
    1  0       0         0         1
    2  0       1         0         0

    >>> import numpy as np
    >>> df = cudf.DataFrame({"a":cudf.Series([1, 2, np.nan, None],
    ...                     nan_as_null=False)})
    >>> df
          a
    0   1.0
    1   2.0
    2   NaN
    3  null

    >>> cudf.get_dummies(df, dummy_na=True, columns=["a"])
       a_1.0  a_2.0  a_nan  a_null
    0      1      0      0       0
    1      0      1      0       0
    2      0      0      1       0
    3      0      0      0       1
    """
    if cats is None:
        cats = {}
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

        if cudf.utils.dtypes.is_list_like(obj):
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
        result_df = df.drop(columns=columns)
        for name in columns:
            if isinstance(
                df[name]._column, cudf.core.column.CategoricalColumn
            ):
                unique = df[name]._column.categories
            else:
                unique = df[name].unique()

            if not dummy_na:
                if np.issubdtype(unique.dtype, np.floating):
                    unique = unique.nans_to_nulls()
                unique = unique.dropna()

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
    result._postprocess_columns(objs[0])
    return result


def _pivot(df, index, columns):
    """
    Reorganize the values of the DataFrame according to the given
    index and columns.

    Parameters
    ----------
    df : DataFrame
    index : cudf.core.index.Index
        Index labels of the result
    columns : cudf.core.index.Index
        Column labels of the result
    """
    columns_labels, columns_idx = columns._encode()
    index_labels, index_idx = index._encode()
    column_labels = columns_labels.to_pandas().to_flat_index()

    # the result of pivot always has a multicolumn
    result = cudf.core.column_accessor.ColumnAccessor(
        multiindex=True, level_names=(None,) + columns._data.names
    )

    def as_tuple(x):
        return x if isinstance(x, tuple) else (x,)

    for v in df:
        names = [as_tuple(v) + as_tuple(name) for name in column_labels]
        col = df._data[v]
        result.update(
            cudf.DataFrame._from_table(
                col.scatter_to_table(
                    index_idx,
                    columns_idx,
                    names,
                    nrows=len(index_labels),
                    ncols=len(names),
                )
            )._data
        )

    return cudf.DataFrame(
        result, index=cudf.Index(index_labels, name=index.name)
    )


def pivot(data, index=None, columns=None, values=None):
    """
    Return reshaped DataFrame organized by the given index and column values.

    Reshape data (produce a "pivot" table) based on column values. Uses
    unique values from specified `index` / `columns` to form axes of the
    resulting DataFrame.

    Parameters
    ----------
    index : column name, optional
        Column used to construct the index of the result.
    columns : column name, optional
        Column used to construct the columns of the result.
    values : column name or list of column names, optional
        Column(s) whose values are rearranged to produce the result.
        If not specified, all remaining columns of the DataFrame
        are used.

    Returns
    -------
    DataFrame

    Examples
    --------
    >>> a = cudf.DataFrame()
    >>> a['a'] = [1, 1, 2, 2],
    >>> a['b'] = ['a', 'b', 'a', 'b']
    >>> a['c'] = [1, 2, 3, 4]
    >>> a.pivot(index='a', columns='b')
       c
    b  a  b
    a
    1  1  2
    2  3  4

    Pivot with missing values in result:

    >>> a = cudf.DataFrame()
    >>> a['a'] = [1, 1, 2]
    >>> a['b'] = [1, 2, 3]
    >>> a['c'] = ['one', 'two', 'three']
    >>> a.pivot(index='a', columns='b')
              c
        b     1     2      3
        a
        1   one   two   <NA>
        2  <NA>  <NA>  three

    """
    df = data
    if values is None:
        values = df._columns_view(
            col for col in df._column_names if col not in (index, columns)
        )
    else:
        values = df._columns_view(values)
    if index is None:
        index = df.index
    else:
        index = cudf.core.index.Index(df.loc[:, index])
    columns = cudf.Index(df.loc[:, columns])

    # Create a DataFrame composed of columns from both
    # columns and index
    columns_index = {}
    columns_index = {
        i: col
        for i, col in enumerate(
            itertools.chain(index._data.columns, columns._data.columns)
        )
    }
    columns_index = cudf.DataFrame(columns_index)

    # Check that each row is unique:
    if len(columns_index) != len(columns_index.drop_duplicates()):
        raise ValueError("Duplicate index-column pairs found. Cannot reshape.")

    return _pivot(values, index, columns)


def unstack(df, level, fill_value=None):
    """
    Pivot one or more levels of the (necessarily hierarchical) index labels.

    Pivots the specified levels of the index labels of df to the innermost
    levels of the columns labels of the result.

    Parameters
    ----------
    df : DataFrame
    level : level name or index, list-like
        Integer, name or list of such, specifying one or more
        levels of the index to pivot
    fill_value
        Non-functional argument provided for compatibility with Pandas.

    Returns
    -------
    DataFrame with specified index levels pivoted to column levels

    Examples
    --------
    >>> df['a'] = [1, 1, 1, 2, 2]
    >>> df['b'] = [1, 2, 3, 1, 2]
    >>> df['c'] = [5, 6, 7, 8, 9]
    >>> df['d'] = ['a', 'b', 'a', 'd', 'e']
    >>> df = df.set_index(['a', 'b', 'd'])
    >>> df
           c
    a b d
    1 1 a  5
      2 b  6
      3 a  7
    2 1 d  8
      2 e  9

    Unstacking level 'a':

    >>> df.unstack('a')
            c
    a       1     2
    b d
    1 a     5  <NA>
      d  <NA>     8
    2 b     6  <NA>
      e  <NA>     9
    3 a     7  <NA>

    Unstacking level 'd' :

    >>> df.unstack('d')
            c
    d       a     b     d     e
    a b
    1 1     5  <NA>  <NA>  <NA>
      2  <NA>     6  <NA>  <NA>
      3     7  <NA>  <NA>  <NA>
    2 1  <NA>  <NA>     8  <NA>
      2  <NA>  <NA>  <NA>     9

    Unstacking multiple levels:

    >>> df.unstack(['b', 'd'])
          c
    b     1           2           3
    d     a     d     b     e     a
    a
    1     5  <NA>     6  <NA>     7
    2  <NA>     8  <NA>     9  <NA>
    """
    if fill_value is not None:
        raise NotImplementedError("fill_value is not supported.")
    if pd.api.types.is_list_like(level):
        if not level:
            return df
    df = df.copy(deep=False)
    if not isinstance(df.index, cudf.MultiIndex):
        raise NotImplementedError(
            "Calling unstack() on a DataFrame without a MultiIndex "
            "is not supported"
        )
    else:
        columns = df.index._poplevels(level)
        index = df.index
    result = _pivot(df, index, columns)
    if result.index.nlevels == 1:
        result.index = result.index.get_level_values(result.index.names[0])
    return result
