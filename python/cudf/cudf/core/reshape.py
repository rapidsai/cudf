# Copyright (c) 2018-2024, NVIDIA CORPORATION.

import itertools
import warnings
from typing import Dict, Optional

import numpy as np
import pandas as pd

import cudf
from cudf._lib.transform import one_hot_encode
from cudf._lib.types import size_type_dtype
from cudf._typing import Dtype
from cudf.api.extensions import no_default
from cudf.core._compat import PANDAS_LT_300
from cudf.core.column import ColumnBase, as_column, column_empty_like
from cudf.core.column.categorical import CategoricalColumn
from cudf.utils.dtypes import min_unsigned_type

_AXIS_MAP = {0: 0, 1: 1, "index": 0, "columns": 1}


def _align_objs(objs, how="outer", sort=None):
    """
    Align a set of Series or Dataframe objects.

    Parameters
    ----------
    objs : list of DataFrame, Series, or Index
    how : How to handle indexes on other axis (or axes),
    similar to join in concat
    sort : Whether to sort the resulting Index

    Returns
    -------
    A list of reindexed and aligned objects
    ready for concatenation
    """
    # Check if multiindex then check if indexes match. Index
    # returns ndarray tuple of bools requiring additional filter.
    # Then check for duplicate index value.
    i_objs = iter(objs)
    first = next(i_objs)

    not_matching_index = any(
        not first.index.equals(rest.index) for rest in i_objs
    )

    if not_matching_index:
        if not all(o.index.is_unique for o in objs):
            raise ValueError("cannot reindex on an axis with duplicate labels")

        index = objs[0].index
        name = index.name

        final_index = _get_combined_index(
            [obj.index for obj in objs], intersect=how == "inner", sort=sort
        )

        final_index.name = name
        return [
            obj.reindex(final_index)
            if not final_index.equals(obj.index)
            else obj
            for obj in objs
        ]
    else:
        if sort:
            if not first.index.is_monotonic_increasing:
                final_index = first.index.sort_values()
                return [obj.reindex(final_index) for obj in objs]
        return objs


def _get_combined_index(indexes, intersect: bool = False, sort=None):
    if len(indexes) == 0:
        index = cudf.Index([])
    elif len(indexes) == 1:
        index = indexes[0]
    elif intersect:
        sort = True
        index = indexes[0]
        for other in indexes[1:]:
            # Don't sort for every intersection,
            # let the sorting happen in the end.
            index = index.intersection(other, sort=False)
    else:
        index = indexes[0]
        if sort is None:
            sort = not index._is_object()
        for other in indexes[1:]:
            index = index.union(other, sort=False)

    if sort:
        if not index.is_monotonic_increasing:
            index = index.sort_values()

    return index


def _normalize_series_and_dataframe(objs, axis):
    """Convert any cudf.Series objects in objs to DataFrames in place."""
    # Default to naming series by a numerical id if they are not named.
    sr_name = 0
    for idx, obj in enumerate(objs):
        if isinstance(obj, cudf.Series):
            name = obj.name
            if name is None:
                if axis == 0:
                    name = 0
                else:
                    name = sr_name
                    sr_name += 1

            objs[idx] = obj.to_frame(name=name)


def concat(objs, axis=0, join="outer", ignore_index=False, sort=None):
    """Concatenate DataFrames, Series, or Indices row-wise.

    Parameters
    ----------
    objs : list or dictionary of DataFrame, Series, or Index
    axis : {0/'index', 1/'columns'}, default 0
        The axis to concatenate along.
        `axis=1` must be passed if a dictionary is passed.
    join : {'inner', 'outer'}, default 'outer'
        How to handle indexes on other axis (or axes).
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
    0      a       1   <NA>
    1      b       2   <NA>
    0      c       3    cat
    1      d       4    dog

    Combine ``DataFrame`` objects with overlapping columns
    and return only those that are shared by passing ``inner`` to
    the ``join`` keyword argument.

    >>> cudf.concat([df1, df3], join="inner")
      letter  number
    0      a       1
    1      b       2
    0      c       3
    1      d       4

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

    Combine a dictionary of DataFrame objects horizontally:

    >>> d = {'first': df1, 'second': df2}
    >>> cudf.concat(d, axis=1)
      first           second
      letter  number  letter  number
    0      a       1       c       3
    1      b       2       d       4
    """
    # TODO: Do we really need to have different error messages for an empty
    # list and a list of None?
    if not objs:
        raise ValueError("No objects to concatenate")

    axis = _AXIS_MAP.get(axis, None)
    if axis is None:
        raise ValueError(
            f'`axis` must be 0 / "index" or 1 / "columns", got: {axis}'
        )

    if isinstance(objs, dict):
        if axis != 1:
            raise NotImplementedError(
                f"Can only concatenate dictionary input along axis=1, not {axis}"
            )
        objs = {k: obj for k, obj in objs.items() if obj is not None}
        keys = list(objs)
        objs = list(objs.values())
        if any(isinstance(o, cudf.BaseIndex) for o in objs):
            raise TypeError(
                "cannot concatenate a dictionary containing indices"
            )
    else:
        objs = [obj for obj in objs if obj is not None]
        keys = None

    if not objs:
        raise ValueError("All objects passed were None")

    # Retrieve the base types of `objs`. In order to support sub-types
    # and object wrappers, we use `isinstance()` instead of comparing
    # types directly
    allowed_typs = {
        cudf.Series,
        cudf.DataFrame,
        cudf.BaseIndex,
    }
    if not all(isinstance(o, tuple(allowed_typs)) for o in objs):
        raise TypeError(
            f"can only concatenate objects which are instances of "
            f"{allowed_typs}, instead received {[type(o) for o in objs]}"
        )

    if any(isinstance(o, cudf.BaseIndex) for o in objs):
        if not all(isinstance(o, cudf.BaseIndex) for o in objs):
            raise TypeError(
                "when concatenating indices you must provide ONLY indices"
            )

    only_series = all(isinstance(o, cudf.Series) for o in objs)

    # Return for single object
    if len(objs) == 1:
        obj = objs[0]
        if ignore_index:
            if axis == 1:
                result = cudf.DataFrame._from_data(
                    data=obj._data.copy(deep=True),
                    index=obj.index.copy(deep=True),
                )
                # The DataFrame constructor for dict-like data (such as the
                # ColumnAccessor given by obj._data here) will drop any columns
                # in the data that are not in `columns`, so we have to rename
                # after construction.
                result.columns = pd.RangeIndex(len(obj._data.names))
            else:
                if isinstance(obj, cudf.Series):
                    result = cudf.Series._from_data(
                        data=obj._data.copy(deep=True),
                        index=cudf.RangeIndex(len(obj)),
                    )
                elif isinstance(obj, pd.Series):
                    result = cudf.Series(
                        data=obj,
                        index=cudf.RangeIndex(len(obj)),
                    )
                else:
                    result = cudf.DataFrame._from_data(
                        data=obj._data.copy(deep=True),
                        index=cudf.RangeIndex(len(obj)),
                    )
        else:
            if axis == 0:
                result = obj.copy()
            else:
                data = obj._data.copy(deep=True)
                if isinstance(obj, cudf.Series) and obj.name is None:
                    # If the Series has no name, pandas renames it to 0.
                    data[0] = data.pop(None)
                result = cudf.DataFrame._from_data(
                    data, index=obj.index.copy(deep=True)
                )
                if keys is not None:
                    if isinstance(result, cudf.DataFrame):
                        k = keys[0]
                        result.columns = cudf.MultiIndex.from_tuples(
                            [
                                (k, *c) if isinstance(c, tuple) else (k, c)
                                for c in result._column_names
                            ]
                        )

        if isinstance(result, cudf.Series) and axis == 0:
            # sort has no effect for series concatted along axis 0
            return result
        else:
            return result.sort_index(axis=(1 - axis)) if sort else result

    # when axis is 1 (column) we can concat with Series and Dataframes
    if axis == 1:
        if not all(isinstance(o, (cudf.Series, cudf.DataFrame)) for o in objs):
            raise TypeError(
                "Can only concatenate Series and DataFrame objects when axis=1"
            )
        df = cudf.DataFrame()
        _normalize_series_and_dataframe(objs, axis=axis)

        any_empty = any(obj.empty for obj in objs)
        if any_empty:
            # Do not remove until pandas-3.0 support is added.
            assert (
                PANDAS_LT_300
            ), "Need to drop after pandas-3.0 support is added."
            warnings.warn(
                "The behavior of array concatenation with empty entries is "
                "deprecated. In a future version, this will no longer exclude "
                "empty items when determining the result dtype. "
                "To retain the old behavior, exclude the empty entries before "
                "the concat operation.",
                FutureWarning,
            )
        # Inner joins involving empty data frames always return empty dfs, but
        # We must delay returning until we have set the column names.
        empty_inner = any_empty and join == "inner"

        objs = [obj for obj in objs if obj.shape != (0, 0)]

        if len(objs) == 0:
            return df

        # Don't need to align indices of all `objs` since we
        # would anyway return an empty dataframe below
        if not empty_inner:
            objs = _align_objs(objs, how=join, sort=sort)
            df.index = objs[0].index

        if keys is None:
            for o in objs:
                for name, col in o._data.items():
                    if name in df._data:
                        raise NotImplementedError(
                            f"A Column with duplicate name found: {name}, cuDF "
                            f"doesn't support having multiple columns with "
                            f"same names yet."
                        )
                    if empty_inner:
                        # if join is inner and it contains an empty df
                        # we return an empty df, hence creating an empty
                        # column with dtype metadata retained.
                        df[name] = cudf.core.column.column_empty_like(
                            col, newsize=0
                        )
                    else:
                        df[name] = col

            result_columns = (
                objs[0]
                ._data.to_pandas_index()
                .append([obj._data.to_pandas_index() for obj in objs[1:]])
                .unique()
            )

        # need to create a MultiIndex column
        else:
            # All levels in the multiindex label must have the same type
            has_multiple_level_types = (
                len({type(name) for o in objs for name in o._data.keys()}) > 1
            )
            if has_multiple_level_types:
                raise NotImplementedError(
                    "Cannot construct a MultiIndex column with multiple "
                    "label types in cuDF at this time. You must convert "
                    "the labels to the same type."
                )
            for k, o in zip(keys, objs):
                for name, col in o._data.items():
                    # if only series, then only keep keys as column labels
                    # if the existing column is multiindex, prepend it
                    # to handle cases where dfs and srs are concatenated
                    if only_series:
                        col_label = k
                    elif isinstance(name, tuple):
                        col_label = (k, *name)
                    else:
                        col_label = (k, name)
                    if empty_inner:
                        df[col_label] = cudf.core.column.column_empty_like(
                            col, newsize=0
                        )
                    else:
                        df[col_label] = col

        if keys is None:
            df.columns = result_columns.unique()
            if ignore_index:
                df.columns = cudf.RangeIndex(len(result_columns.unique()))
        elif ignore_index:
            # with ignore_index the column names change to numbers
            df.columns = cudf.RangeIndex(len(result_columns))
        elif not only_series:
            df.columns = cudf.MultiIndex.from_tuples(df._column_names)

        if empty_inner:
            # if join is inner and it contains an empty df
            # we return an empty df
            return df.head(0)

        return df

    # If we get here, we are always concatenating along axis 0 (the rows).
    typ = type(objs[0])
    if len({type(o) for o in objs}) > 1:
        _normalize_series_and_dataframe(objs, axis=axis)
        typ = cudf.DataFrame

    if typ is cudf.DataFrame:
        old_objs = objs
        objs = [obj for obj in objs if obj.shape != (0, 0)]
        if len(objs) == 0:
            # If objs is empty, that indicates all of
            # objs are empty dataframes.
            return cudf.DataFrame()
        elif len(objs) == 1:
            obj = objs[0]
            result = cudf.DataFrame._from_data(
                data=None if join == "inner" else obj._data.copy(deep=True),
                index=cudf.RangeIndex(len(obj))
                if ignore_index
                else obj.index.copy(deep=True),
            )
            return result
        else:
            if join == "inner" and len(old_objs) != len(objs):
                # don't filter out empty df's
                objs = old_objs
            result = cudf.DataFrame._concat(
                objs,
                axis=axis,
                join=join,
                ignore_index=ignore_index,
                # Explicitly cast rather than relying on None being falsy.
                sort=bool(sort),
            )
        return result

    elif typ is cudf.Series:
        new_objs = [obj for obj in objs if len(obj)]
        if len(new_objs) == 1 and not ignore_index:
            return new_objs[0]
        else:
            return cudf.Series._concat(
                objs, axis=axis, index=None if ignore_index else True
            )
    elif typ is cudf.MultiIndex:
        return cudf.MultiIndex._concat(objs)
    elif issubclass(typ, cudf.Index):
        return cudf.core.index.Index._concat(objs)
    else:
        raise TypeError(f"cannot concatenate object of type {typ}")


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

    The names of 'variable' and 'value' columns can be customized:

    >>> cudf.melt(df, id_vars=['A'], value_vars=['B'],
    ...         var_name='myVarname', value_name='myValname')
       A myVarname  myValname
    0  a         B          1
    1  b         B          3
    2  c         B          5
    """
    if col_level is not None:
        raise NotImplementedError("col_level != None is not supported yet.")

    # Arg cleaning

    # id_vars
    if id_vars is not None:
        if cudf.api.types.is_scalar(id_vars):
            id_vars = [id_vars]
        id_vars = list(id_vars)
        missing = set(id_vars) - set(frame._column_names)
        if not len(missing) == 0:
            raise KeyError(
                f"The following 'id_vars' are not present"
                f" in the DataFrame: {list(missing)}"
            )
    else:
        id_vars = []

    # value_vars
    if value_vars is not None:
        if cudf.api.types.is_scalar(value_vars):
            value_vars = [value_vars]
        value_vars = list(value_vars)
        missing = set(value_vars) - set(frame._column_names)
        if not len(missing) == 0:
            raise KeyError(
                f"The following 'value_vars' are not present"
                f" in the DataFrame: {list(missing)}"
            )
    else:
        # then all remaining columns in frame
        unique_id = set(id_vars)
        value_vars = [c for c in frame._column_names if c not in unique_id]

    # Error for unimplemented support for datatype
    dtypes = [frame[col].dtype for col in id_vars + value_vars]
    if any(isinstance(typ, cudf.CategoricalDtype) for typ in dtypes):
        raise NotImplementedError(
            "Categorical columns are not yet supported for function"
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
            f"'value_vars' and 'id_vars' cannot have overlap."
            f" The following 'value_vars' are ALSO present"
            f" in 'id_vars': {list(overlap)}"
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
    mdata = {col: _tile(frame[col], K) for col in id_vars}

    # Step 2: add variable
    nval = len(value_vars)
    dtype = min_unsigned_type(nval)

    if not var_name:
        var_name = "variable"

    if not value_vars:
        # TODO: Use frame._data.label_dtype when it's more consistently set
        var_data = cudf.Series(
            value_vars, dtype=frame._data.to_pandas_index().dtype
        )
    else:
        var_data = (
            cudf.Series(value_vars)
            .take(np.repeat(np.arange(nval, dtype=dtype), N))
            .reset_index(drop=True)
        )
    mdata[var_name] = var_data

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
    dtype="bool",
):
    """Returns a dataframe whose columns are the one hot encodings of all
    columns in `df`

    Parameters
    ----------
    df : array-like, Series, or DataFrame
        Data of which to get dummy indicators.
    prefix : str, dict, or sequence, optional
        Prefix to append. Either a str (to apply a constant prefix), dict
        mapping column names to prefixes, or sequence of prefixes to apply with
        the same length as the number of columns. If not supplied, defaults
        to the empty string
    prefix_sep : str, dict, or sequence, optional, default '_'
        Separator to use when appending prefixes
    dummy_na : boolean, optional
        Add a column to indicate Nones, if False Nones are ignored.
    cats : dict, optional
        Dictionary mapping column names to sequences of values representing
        that column's category. If not supplied, it is computed as the unique
        values of the column.
    sparse : boolean, optional
        Right now this is NON-FUNCTIONAL argument in rapids.
    drop_first : boolean, optional
        Right now this is NON-FUNCTIONAL argument in rapids.
    columns : sequence of str, optional
        Names of columns to encode. If not provided, will attempt to encode all
        columns. Note this is different from pandas default behavior, which
        encodes all columns with dtype object or categorical
    dtype : str, optional
        Output dtype, default 'bool'

    Examples
    --------
    >>> import cudf
    >>> df = cudf.DataFrame({"a": ["value1", "value2", None], "b": [0, 0, 0]})
    >>> cudf.get_dummies(df)
       b  a_value1  a_value2
    0  0      True     False
    1  0     False      True
    2  0     False     False

    >>> cudf.get_dummies(df, dummy_na=True)
       b  a_<NA>  a_value1  a_value2
    0  0   False      True     False
    1  0   False     False      True
    2  0    True     False     False

    >>> import numpy as np
    >>> df = cudf.DataFrame({"a":cudf.Series([1, 2, np.nan, None],
    ...                     nan_as_null=False)})
    >>> df
          a
    0   1.0
    1   2.0
    2   NaN
    3  <NA>

    >>> cudf.get_dummies(df, dummy_na=True, columns=["a"])
       a_<NA>  a_1.0  a_2.0  a_nan
    0   False   True  False  False
    1   False  False   True  False
    2   False  False  False   True
    3    True  False  False  False

    >>> series = cudf.Series([1, 2, None, 2, 4])
    >>> series
    0       1
    1       2
    2    <NA>
    3       2
    4       4
    dtype: int64
    >>> cudf.get_dummies(series, dummy_na=True)
        <NA>      1      2      4
    0  False   True  False  False
    1  False  False   True  False
    2   True  False  False  False
    3  False  False   True  False
    4  False  False  False   True
    """

    if cats is None:
        cats = {}
    if sparse:
        raise NotImplementedError("sparse is not supported yet")

    if drop_first:
        raise NotImplementedError("drop_first is not supported yet")

    if isinstance(df, cudf.DataFrame):
        encode_fallback_dtypes = ["object", "category"]

        if columns is None or len(columns) == 0:
            columns = df.select_dtypes(
                include=encode_fallback_dtypes
            )._column_names

        _length_check_params(prefix, columns, "prefix")
        _length_check_params(prefix_sep, columns, "prefix_sep")

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

        # If we have no columns to encode, we need to drop
        # fallback columns(if any)
        if len(columns) == 0:
            return df.select_dtypes(exclude=encode_fallback_dtypes)
        else:
            result_data = {
                col_name: col
                for col_name, col in df._data.items()
                if col_name not in columns
            }

            for name in columns:
                if name not in cats:
                    unique = _get_unique(
                        column=df._data[name], dummy_na=dummy_na
                    )
                else:
                    unique = as_column(cats[name])

                col_enc_data = _one_hot_encode_column(
                    column=df._data[name],
                    categories=unique,
                    prefix=prefix_map.get(name, prefix),
                    prefix_sep=prefix_sep_map.get(name, prefix_sep),
                    dtype=dtype,
                )
                result_data.update(col_enc_data)
            return cudf.DataFrame._from_data(result_data, index=df._index)
    else:
        ser = cudf.Series(df)
        unique = _get_unique(column=ser._column, dummy_na=dummy_na)
        data = _one_hot_encode_column(
            column=ser._column,
            categories=unique,
            prefix=prefix,
            prefix_sep=prefix_sep,
            dtype=dtype,
        )
        return cudf.DataFrame._from_data(data, index=ser._index)


def _merge_sorted(
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
    objs : list of DataFrame or Series
    keys : list, default None
        List of Column names to sort by. If None, all columns used
        (Ignored if `by_index=True`)
    by_index : bool, default False
        Use index for sorting. `keys` input will be ignored if True
    ignore_index : bool, default False
        Drop and ignore index during merge. Default range index will
        be used in the output dataframe.
    ascending : bool, default True
        Sorting is in ascending order, otherwise it is descending
    na_position : {'first', 'last'}, default 'last'
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

    if by_index:
        key_columns_indices = list(range(0, objs[0]._index.nlevels))
    else:
        if keys is None:
            key_columns_indices = list(range(0, objs[0]._num_columns))
        else:
            key_columns_indices = [
                objs[0]._column_names.index(key) for key in keys
            ]
        if not ignore_index:
            key_columns_indices = [
                idx + objs[0]._index.nlevels for idx in key_columns_indices
            ]

    columns = [
        [
            *(obj._index._data.columns if not ignore_index else ()),
            *obj._columns,
        ]
        for obj in objs
    ]

    return objs[0]._from_columns_like_self(
        cudf._lib.merge.merge_sorted(
            input_columns=columns,
            key_columns_indices=key_columns_indices,
            ascending=ascending,
            na_position=na_position,
        ),
        column_names=objs[0]._column_names,
        index_names=None if ignore_index else objs[0]._index_names,
    )


def _pivot(df, index, columns):
    """
    Reorganize the values of the DataFrame according to the given
    index and columns.

    Parameters
    ----------
    df : DataFrame
    index : cudf.Index
        Index labels of the result
    columns : cudf.Index
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
        nrows = len(index_labels)
        ncols = len(names)
        num_elements = nrows * ncols
        if num_elements > 0:
            col = df._data[v]
            scatter_map = (columns_idx * np.int32(nrows)) + index_idx
            target = cudf.DataFrame._from_data(
                {
                    None: cudf.core.column.column_empty_like(
                        col, masked=True, newsize=nrows * ncols
                    )
                }
            )
            target._data[None][scatter_map] = col
            result_frames = target._split(range(nrows, nrows * ncols, nrows))
            result.update(
                {
                    name: next(iter(f._columns))
                    for name, f in zip(names, result_frames)
                }
            )

    return cudf.DataFrame._from_data(
        result, index=cudf.Index(index_labels, name=index.name)
    )


def pivot(data, columns=None, index=no_default, values=no_default):
    """
    Return reshaped DataFrame organized by the given index and column values.

    Reshape data (produce a "pivot" table) based on column values. Uses
    unique values from specified `index` / `columns` to form axes of the
    resulting DataFrame.

    Parameters
    ----------
    columns : column name, optional
        Column used to construct the columns of the result.
    index : column name, optional
        Column used to construct the index of the result.
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
    >>> a['a'] = [1, 1, 2, 2]
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
    values_is_list = True
    if values is no_default:
        values = df._columns_view(
            col for col in df._column_names if col not in (index, columns)
        )
    else:
        if not isinstance(values, (list, tuple)):
            values = [values]
            values_is_list = False
        values = df._columns_view(values)
    if index is no_default:
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

    result = _pivot(values, index, columns)

    # MultiIndex to Index
    if not values_is_list:
        result._data.droplevel(0)

    return result


def unstack(df, level, fill_value=None):
    """
    Pivot one or more levels of the (necessarily hierarchical) index labels.

    Pivots the specified levels of the index labels of df to the innermost
    levels of the columns labels of the result.

    * If the index of ``df`` has multiple levels, returns a ``Dataframe`` with
      specified level of the index pivoted to the column levels.
    * If the index of ``df`` has single level, returns a ``Series`` with all
      column levels pivoted to the index levels.

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
    Series or DataFrame

    Examples
    --------
    >>> df = cudf.DataFrame()
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

    Unstacking single level index dataframe:

    >>> df = cudf.DataFrame({('c', 1): [1, 2, 3], ('c', 2):[9, 8, 7]})
    >>> df.unstack()
    c  1  0    1
          1    2
          2    3
       2  0    9
          1    8
          2    7
    dtype: int64
    """
    if not isinstance(df, cudf.DataFrame):
        raise ValueError("`df` should be a cudf Dataframe object.")

    if df.empty:
        raise ValueError("Cannot unstack an empty dataframe.")

    if fill_value is not None:
        raise NotImplementedError("fill_value is not supported.")
    if pd.api.types.is_list_like(level):
        if not level:
            return df
    df = df.copy(deep=False)
    if not isinstance(df.index, cudf.MultiIndex):
        dtype = df._columns[0].dtype
        for col in df._columns:
            if not col.dtype == dtype:
                raise ValueError(
                    "Calling unstack() on single index dataframe"
                    " with different column datatype is not supported."
                )
        res = df.T.stack(future_stack=False)
        # Result's index is a multiindex
        res.index.names = (
            tuple(df._data.to_pandas_index().names) + df.index.names
        )
        return res
    else:
        columns = df.index._poplevels(level)
        index = df.index
    result = _pivot(df, index, columns)
    if result.index.nlevels == 1:
        result.index = result.index.get_level_values(result.index.names[0])
    return result


def _get_unique(column, dummy_na):
    """
    Returns unique values in a column, if
    dummy_na is False, nan's are also dropped.
    """
    if isinstance(column, cudf.core.column.CategoricalColumn):
        unique = column.categories
    else:
        unique = column.unique().sort_values()
    if not dummy_na:
        if np.issubdtype(unique.dtype, np.floating):
            unique = unique.nans_to_nulls()
        unique = unique.dropna()
    return unique


def _one_hot_encode_column(
    column: ColumnBase,
    categories: ColumnBase,
    prefix: Optional[str],
    prefix_sep: Optional[str],
    dtype: Optional[Dtype],
) -> Dict[str, ColumnBase]:
    """Encode a single column with one hot encoding. The return dictionary
    contains pairs of (category, encodings). The keys may be prefixed with
    `prefix`, separated with category name with `prefix_sep`. The encoding
    columns maybe coerced into `dtype`.
    """
    if isinstance(column, CategoricalColumn):
        if column.size == column.null_count:
            column = column_empty_like(categories, newsize=column.size)
        else:
            column = column._get_decategorized_column()

    if column.size * categories.size >= np.iinfo(size_type_dtype).max:
        raise ValueError(
            "Size limitation exceeded: column.size * category.size < "
            f"np.iinfo({size_type_dtype}).max. Consider reducing "
            "size of category"
        )
    data = one_hot_encode(column, categories)

    if prefix is not None and prefix_sep is not None:
        data = {f"{prefix}{prefix_sep}{col}": enc for col, enc in data.items()}
    if dtype:
        data = {k: v.astype(dtype) for k, v in data.items()}
    return data


def _length_check_params(obj, columns, name):
    if cudf.api.types.is_list_like(obj):
        if len(obj) != len(columns):
            raise ValueError(
                f"Length of '{name}' ({len(obj)}) did not match the "
                f"length of the columns being "
                f"encoded ({len(columns)})."
            )


def _get_pivot_names(arrs, names, prefix):
    """
    Generates unique names for rows/columns
    """
    if names is None:
        names = []
        for i, arr in enumerate(arrs):
            if isinstance(arr, cudf.Series) and arr.name is not None:
                names.append(arr.name)
            else:
                names.append(f"{prefix}_{i}")
    else:
        if len(names) != len(arrs):
            raise ValueError("arrays and names must have the same length")
        if not isinstance(names, list):
            names = list(names)

    return names


def crosstab(
    index,
    columns,
    values=None,
    rownames=None,
    colnames=None,
    aggfunc=None,
    margins=False,
    margins_name="All",
    dropna=None,
    normalize=False,
):
    """
    Compute a simple cross tabulation of two (or more) factors. By default
    computes a frequency table of the factors unless an array of values and an
    aggregation function are passed.

    Parameters
    ----------
    index : array-like, Series, or list of arrays/Series
        Values to group by in the rows.
    columns : array-like, Series, or list of arrays/Series
        Values to group by in the columns.
    values : array-like, optional
        Array of values to aggregate according to the factors.
        Requires `aggfunc` be specified.
    rownames : list of str, default None
        If passed, must match number of row arrays passed.
    colnames : list of str, default None
        If passed, must match number of column arrays passed.
    aggfunc : function, optional
        If specified, requires `values` be specified as well.
    margins : Not supported
    margins_name : Not supported
    dropna : Not supported
    normalize : Not supported

    Returns
    -------
    DataFrame
        Cross tabulation of the data.

    Examples
    --------
    >>> a = cudf.Series(["foo", "foo", "foo", "foo", "bar", "bar",
    ...               "bar", "bar", "foo", "foo", "foo"], dtype=object)
    >>> b = cudf.Series(["one", "one", "one", "two", "one", "one",
    ...               "one", "two", "two", "two", "one"], dtype=object)
    >>> c = cudf.Series(["dull", "dull", "shiny", "dull", "dull", "shiny",
    ...               "shiny", "dull", "shiny", "shiny", "shiny"],
    ...              dtype=object)
    >>> cudf.crosstab(a, [b, c], rownames=['a'], colnames=['b', 'c'])
    b   one        two
    c   dull shiny dull shiny
    a
    bar    1     2    1     0
    foo    2     2    1     2
    """
    if normalize is not False:
        raise NotImplementedError("normalize is not supported yet")

    if values is None and aggfunc is not None:
        raise ValueError("aggfunc cannot be used without values.")

    if values is not None and aggfunc is None:
        raise ValueError("values cannot be used without an aggfunc.")

    if not isinstance(index, (list, tuple)):
        index = [index]
    if not isinstance(columns, (list, tuple)):
        columns = [columns]

    if not rownames:
        rownames = _get_pivot_names(index, rownames, prefix="row")
    if not colnames:
        colnames = _get_pivot_names(columns, colnames, prefix="col")

    if len(index) != len(rownames):
        raise ValueError("index and rownames must have same length")
    if len(columns) != len(colnames):
        raise ValueError("columns and colnames must have same length")

    if len(set(rownames)) != len(rownames):
        raise ValueError("rownames must be unique")
    if len(set(colnames)) != len(colnames):
        raise ValueError("colnames must be unique")

    data = {
        **dict(zip(rownames, map(as_column, index))),
        **dict(zip(colnames, map(as_column, columns))),
    }

    df = cudf.DataFrame._from_data(data)

    if values is None:
        df["__dummy__"] = 0
        kwargs = {"aggfunc": "count", "fill_value": 0}
    else:
        df["__dummy__"] = values
        kwargs = {"aggfunc": aggfunc}

    table = pivot_table(
        data=df,
        index=rownames,
        columns=colnames,
        values="__dummy__",
        margins=margins,
        margins_name=margins_name,
        dropna=dropna,
        **kwargs,
    )

    return table


def pivot_table(
    data,
    values=None,
    index=None,
    columns=None,
    aggfunc="mean",
    fill_value=None,
    margins=False,
    dropna=None,
    margins_name="All",
    observed=False,
    sort=True,
):
    """
    Create a spreadsheet-style pivot table as a DataFrame.

    Parameters
    ----------
    data : DataFrame
    values : column name or list of column names to aggregate, optional
    index : list of column names
            Values to group by in the rows.
    columns : list of column names
            Values to group by in the columns.
    aggfunc : str or dict, default "mean"
            If dict is passed, the key is column to aggregate
            and value is function name.
    fill_value : scalar, default None
        Value to replace missing values with
        (in the resulting pivot table, after aggregation).
    margins : Not supported
    dropna : Not supported
    margins_name : Not supported
    observed : Not supported
    sort : Not supported

    Returns
    -------
    DataFrame
        An Excel style pivot table.
    """
    if margins is not False:
        raise NotImplementedError("margins is not supported yet")

    if margins_name != "All":
        raise NotImplementedError("margins_name is not supported yet")

    if dropna is not None:
        raise NotImplementedError("dropna is not supported yet")

    if observed is not False:
        raise NotImplementedError("observed is not supported yet")

    if sort is not True:
        raise NotImplementedError("sort is not supported yet")

    keys = index + columns

    values_passed = values is not None
    if values_passed:
        if pd.api.types.is_list_like(values):
            values_multi = True
            values = list(values)
        else:
            values_multi = False
            values = [values]

        for i in values:
            if i not in data:
                raise KeyError(i)

        to_filter = []
        for x in keys + values:
            if isinstance(x, cudf.Grouper):
                x = x.key
            try:
                if x in data:
                    to_filter.append(x)
            except TypeError:
                pass
        if len(to_filter) < len(data._column_names):
            data = data[to_filter]

    else:
        values = data.columns
        for key in keys:
            try:
                values = values.drop(key)
            except (TypeError, ValueError, KeyError):
                pass
        values = list(values)

    grouped = data.groupby(keys)
    agged = grouped.agg(aggfunc)

    table = agged

    if table.index.nlevels > 1 and index:
        # If index_names are integers, determine whether the integers refer
        # to the level position or name.
        index_names = agged.index.names[: len(index)]
        to_unstack = []
        for i in range(len(index), len(keys)):
            name = agged.index.names[i]
            if name is None or name in index_names:
                to_unstack.append(i)
            else:
                to_unstack.append(name)
        table = agged.unstack(to_unstack)

    if fill_value is not None:
        table = table.fillna(fill_value)

    # discard the top level
    if values_passed and not values_multi and table._data.multiindex:
        column_names = table._data.level_names[1:]
        table_columns = tuple(
            map(lambda column: column[1:], table._data.names)
        )
        table.columns = cudf.MultiIndex.from_tuples(
            tuples=table_columns, names=column_names
        )

    if len(index) == 0 and len(columns) > 0:
        table = table.T

    return table
