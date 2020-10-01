# Copyright (c) 2020, NVIDIA CORPORATION.
from typing import Union

import numpy as np
import pandas as pd

import cudf
from cudf.utils.dtypes import is_categorical_dtype


def _check_isinstance(left, right, obj):
    if not isinstance(left, obj):
        raise AssertionError(
            f"{obj} Expected type {obj}, found {type(left)} instead"
        )
    elif not isinstance(right, obj):
        raise AssertionError(
            f"{obj} Expected type {obj}, found {type(right)} instead"
        )


def raise_assert_detail(obj, message, left, right, diff=None):

    msg = f"""{obj} are different

{message}
[left]:  {left}
[right]: {right}"""

    if diff is not None:
        msg += f"\n[diff]: {diff}"

    raise AssertionError(msg)


def _check_types(
    left, right, check_categorical=True, exact="equiv", obj="Index"
):
    if not exact or exact == "equiv":
        if (
            isinstance(left, cudf.RangeIndex)
            and isinstance(
                right,
                (
                    cudf.Int8Index,
                    cudf.Int16Index,
                    cudf.Int32Index,
                    cudf.Int64Index,
                ),
            )
        ) or (
            isinstance(right, cudf.RangeIndex)
            and isinstance(
                left,
                (
                    cudf.Int8Index,
                    cudf.Int16Index,
                    cudf.Int32Index,
                    cudf.Int64Index,
                ),
            )
        ):
            return

    if type(left) != type(right):
        raise_assert_detail(
            obj, "Class types are different", f"{type(left)}", f"{type(right)}"
        )

    if (
        exact
        and not isinstance(left, cudf.MultiIndex)
        and is_categorical_dtype(left)
    ):
        if left.dtype != right.dtype:
            raise_assert_detail(
                obj, "Categorical difference", f"{left}", f"{right}"
            )


def assert_column_equal(
    left,
    right,
    check_dtype=True,
    check_column_type="equiv",
    check_less_precise=False,
    check_exact=False,
    check_datetimelike_compat=False,
    check_categorical=True,
    check_category_order=True,
    obj="ColumnBase",
):
    """ Check that left and right columns are equal

    This function is intended to compare two columns and output
    any differences. Additional parameters allow varying the strictness
    of the equality checks performed.

    Parameters:
    -----------
    left : Column
        left Column to compare
    right : Column
        right Column to compare
    check_dtype : bool, default True
        Whether to check the Column dtype is identical.
    check_column_type : bool or {‘equiv’}, default ‘equiv’
        Whether to check the columns class, dtype and
        inferred_type are identical. Currently it is idle,
        and similar to pandas.
    check_less_precise : bool or int, default False
        Not yet supported
    check_exact : bool, default False
        Whether to compare number exactly.
    check_datetime_like_compat : bool, default False
        Compare datetime-like which is comparable ignoring dtype.
    check_categorical : bool, default True
        Whether to compare internal Categorical exactly.
    check_category_order : bool, default True
        Whether to compare category order of internal Categoricals
    obj : str, default ‘ColumnBase’
        Specify object name being compared, internally used to
        show appropriate assertion message.
    """
    if check_dtype is True:
        if (
            is_categorical_dtype(left)
            and is_categorical_dtype(right)
            and not check_categorical
        ):
            pass
        else:
            if type(left) != type(right) or left.dtype != right.dtype:
                msg1 = f"{left.dtype}"
                msg2 = f"{right.dtype}"
                raise_assert_detail(obj, "Dtypes are different", msg1, msg2)

    if check_datetimelike_compat:
        if np.issubdtype(left.dtype, np.datetime64):
            right = right.astype(left.dtype)
        elif np.issubdtype(right.dtype, np.datetime64):
            left = left.astype(right.dtype)

        if np.issubdtype(left.dtype, np.datetime64):
            if not left.equals(right):
                raise AssertionError(
                    f"[datetimelike_compat=True] {left.values} "
                    f"is not equal to {right.values}."
                )
            return

    if check_exact and check_categorical:
        if is_categorical_dtype(left) and is_categorical_dtype(right):
            left_cat = left.cat().categories
            right_cat = right.cat().categories

            if check_category_order:
                assert_index_equal(
                    left_cat,
                    right_cat,
                    exact=check_dtype,
                    check_exact=True,
                    check_categorical=False,
                )
                assert_column_equal(
                    left.codes,
                    right.codes,
                    check_dtype=check_dtype,
                    check_exact=True,
                    check_categorical=False,
                    check_category_order=False,
                )

            if left.ordered != right.ordered:
                msg1 = f"{left.ordered}"
                msg2 = f"{right.ordered}"
                raise_assert_detail(
                    "{obj} category", "Orders are different", msg1, msg2
                )

    if (
        not check_dtype
        and is_categorical_dtype(left)
        and is_categorical_dtype(right)
    ):
        left = left.astype(left.categories.dtype)
        right = right.astype(right.categories.dtype)

    columns_equal = False
    try:
        columns_equal = left.equals(right)
    except TypeError as e:
        if str(e) != "Categoricals can only compare with the same type":
            raise e
        if is_categorical_dtype(left) and is_categorical_dtype(right):
            left = left.astype(left.categories.dtype)
            right = right.astype(right.categories.dtype)
    if not columns_equal:
        msg1 = f"{left.to_array()}"
        msg2 = f"{right.to_array()}"
        try:
            diff = left.apply_boolean_mask(
                left.binary_operator("ne", right)
            ).size
            diff = diff * 100.0 / left.size
        except BaseException:
            diff = 100.0
        raise_assert_detail(
            obj, f"values are different ({np.round(diff, 5)} %)", msg1, msg2,
        )


def assert_index_equal(
    left,
    right,
    exact="equiv",
    check_names: bool = True,
    check_less_precise: Union[bool, int] = False,
    check_exact: bool = True,
    check_categorical: bool = True,
    obj: str = "Index",
):
    """ Check that left and right Index are equal

    This function is intended to compare two Index and output
    any differences. Additional parameters allow varying the strictness
    of the equality checks performed.

    Parameters:
    -----------
    left : Index
        left Index to compare
    right : Index
        right Index to compare
    exact : bool or {‘equiv’}, default ‘equiv’
        Whether to check the Index class, dtype and inferred_type
        are identical. If ‘equiv’, then RangeIndex can be substituted
        for Int8Index, Int16Index, Int32Index, Int64Index as well.
    check_names : bool, default True
        Whether to check the names attribute.
    check_less_precise : bool or int, default False
        Not yet supported
    check_exact : bool, default False
        Whether to compare number exactly.
    check_categorical : bool, default True
        Whether to compare internal Categorical exactly.
    obj : str, default ‘Index’
        Specify object name being compared, internally used to
        show appropriate assertion message.

    Examples
    --------
    >>> import cudf
    >>> id1 = cudf.Index([1, 2, 3, 4])
    >>> id2 = cudf.Index([1, 2, 3, 5])
    >>> cudf.testing.assert_index_equal(id1, id2)
    ......
    ......
    AssertionError: ColumnBase are different

    values are different (25.0 %)
    [left]:  [1 2 3 4]
    [right]: [1 2 3 5]

    >>> id2 = cudf.Index([1, 2, 3, 4], name="b")
    >>> cudf.testing.assert_index_equal(id1, id2)
    ......
    ......
    AssertionError: Index are different

    name mismatch
    [left]:  a
    [right]: b

    # This will pass without any hitch
    >>> id2 = cudf.Index([1, 2, 3, 4], name="a")
    >>> cudf.testing.assert_index_equal(id1, id2)
    """

    # instance validation
    _check_isinstance(left, right, cudf.Index)

    _check_types(
        left, right, exact=exact, check_categorical=check_categorical, obj=obj
    )

    if len(left) != len(right):
        raise_assert_detail(
            obj, "lengths are different", f"{len(left)}", f"{len(right)}"
        )

    if isinstance(left, cudf.MultiIndex):
        if left.nlevels != right.nlevels:
            raise AssertionError(
                "Number of levels mismatch, "
                f"left has {left.nlevels} levels and right has {right.nlevels}"
            )

        for level in range(left.nlevels):
            llevel = cudf.Index(left._columns[level], name=left.names[level])
            rlevel = cudf.Index(right._columns[level], name=right.names[level])
            mul_obj = f"MultiIndex level [{level}]"
            assert_index_equal(
                llevel,
                rlevel,
                exact=check_exact,
                check_names=check_names,
                check_less_precise=check_less_precise,
                check_exact=check_exact,
                obj=mul_obj,
            )
        return

    assert_column_equal(
        left._columns[0],
        right._columns[0],
        check_dtype=exact,
        check_exact=check_exact,
        check_categorical=check_categorical,
        obj=obj,
    )

    # metadata comparison
    if check_names and (left.name != right.name):
        raise_assert_detail(
            obj, "name mismatch", f"{left.name}", f"{right.name}"
        )


def assert_series_equal(
    left,
    right,
    check_dtype=True,
    check_index_type="equiv",
    check_series_type=True,
    check_less_precise=False,
    check_names=True,
    check_exact=False,
    check_datetimelike_compat=False,
    check_categorical=True,
    check_category_order=True,
    obj="Series",
):
    """ Check that left and right Series are equal

    This function is intended to compare two Series and output
    any differences. Additional parameters allow varying the strictness
    of the equality checks performed.

    Parameters:
    -----------
    left : Series
        left Series to compare
    right : Series
        right Series to compare
    check_dtype : bool, default True
        Whether to check the Series dtype is identical.
    check_index_type : bool or {‘equiv’}, default ‘equiv’
        Whether to check the Index class, dtype and inferred_type
        are identical.
    check_series_type : bool, default True
        Whether to check the seires class, dtype and
        inferred_type are identical. Currently it is idle,
        and similar to pandas.
    check_less_precise : bool or int, default False
        Not yet supported
    check_names : bool, default True
        Whether to check that the names attribute for both the index
        and column attributes of the Series is identical.
    check_exact : bool, default False
        Whether to compare number exactly.
    check_datetime_like_compat : bool, default False
        Compare datetime-like which is comparable ignoring dtype.
    check_categorical : bool, default True
        Whether to compare internal Categorical exactly.
    check_category_order : bool, default True
        Whether to compare category order of internal Categoricals
    obj : str, default ‘Series’
        Specify object name being compared, internally used to
        show appropriate assertion message.

    Examples
    --------
    >>> import cudf
    >>> sr1 = cudf.Series([1, 2, 3, 4], name="a")
    >>> sr2 = cudf.Series([1, 2, 3, 5], name="b")
    >>> cudf.testing.assert_series_equal(sr1, sr2)
    ......
    ......
    AssertionError: ColumnBase are different

    values are different (25.0 %)
    [left]:  [1 2 3 4]
    [right]: [1 2 3 5]

    >>> sr2 = cudf.Series([1, 2, 3, 4], name="b")
    >>> cudf.testing.assert_series_equal(sr1, sr2)
    ......
    ......
    AssertionError: Series are different

    name mismatch
    [left]:  a
    [right]: b

    # This will pass without any hitch
    >>> sr2 = cudf.Series([1, 2, 3, 4], name="a")
    >>> cudf.testing.assert_series_equal(sr1, sr2)
    """

    # instance validation
    _check_isinstance(left, right, cudf.Series)

    if len(left) != len(right):
        msg1 = f"{len(left)}, {left.index}"
        msg2 = f"{len(right)}, {right.index}"
        raise_assert_detail(obj, "Series length are different", msg1, msg2)

    # index comparison
    assert_index_equal(
        left.index,
        right.index,
        exact=check_index_type,
        check_names=check_names,
        check_less_precise=check_less_precise,
        check_exact=check_exact,
        check_categorical=check_categorical,
        obj=f"{obj}.index",
    )

    assert_column_equal(
        left._column,
        right._column,
        check_dtype=check_dtype,
        check_column_type=check_series_type,
        check_less_precise=check_less_precise,
        check_exact=check_exact,
        check_datetimelike_compat=check_datetimelike_compat,
        check_categorical=check_categorical,
        check_category_order=check_category_order,
    )

    # metadata comparison
    if check_names and (left.name != right.name):
        raise_assert_detail(
            obj, "name mismatch", f"{left.name}", f"{right.name}"
        )


def assert_frame_equal(
    left,
    right,
    check_dtype=True,
    check_index_type="equiv",
    check_column_type="equiv",
    check_frame_type=True,
    check_less_precise=False,
    by_blocks=False,
    check_names=True,
    check_exact=False,
    check_datetimelike_compat=False,
    check_categorical=True,
    check_like=False,
    obj="DataFrame",
):
    """ Check that left and right DataFrame are equal

    This function is intended to compare two DataFrame and output
    any differences. Additional parameters allow varying the strictness
    of the equality checks performed.

    Parameters:
    -----------
    left : DataFrame
        left DataFrame to compare
    right : DataFrame
        right DataFrame to compare
    check_dtype : bool, default True
        Whether to check the DataFrame dtype is identical.
    check_index_type : bool or {‘equiv’}, default ‘equiv’
        Whether to check the Index class, dtype and inferred_type
        are identical.
    check_column_type : bool, default True
        Whether to check the column class, dtype and
        inferred_type are identical. Currently it is idle,
        and similar to pandas.
    check_frame_type : bool, default True
        Whether to check the DataFrame class is identical.
    check_less_precise : bool or int, default False
        Not yet supported
    check_names : bool, default True
        Whether to check that the names attribute for both the index and
        column attributes of the DataFrame is identical.
    check_exact : bool, default False
        Whether to compare number exactly.
    by_blocks : bool, default False
        Not supported
    check_exact : bool, default False
        Whether to compare number exactly.
    check_datetime_like_compat : bool, default False
        Compare datetime-like which is comparable ignoring dtype.
    check_categorical : bool, default True
        Whether to compare internal Categorical exactly.
    check_like : bool, default False
        If True, ignore the order of index & columns.
        Note: index labels must match their respective
        rows (same as in columns) - same labels must be with the same data.
    obj : str, default ‘DataFrame’
        Specify object name being compared, internally used to
        show appropriate assertion message.

    Examples
    --------
    >>> import cudf
    >>> df1 = cudf.DataFrame({"a":[1, 2], "b":[1.0, 2.0]}, index=[1, 2])
    >>> df2 = cudf.DataFrame({"a":[1, 2], "b":[1.0, 2.0]}, index=[2, 3])
    >>> cudf.testing.assert_frame_equal(df1, df2)
    ......
    ......
    AssertionError: ColumnBase are different

    values are different (100.0 %)
    [left]:  [1 2]
    [right]: [2 3]

    >>> df2 = cudf.DataFrame({"a":[1, 2], "c":[1.0, 2.0]}, index=[1, 2])
    >>> cudf.testing.assert_frame_equal(df1, df2)
    ......
    ......
    AssertionError: DataFrame.columns are different

    DataFrame.columns values are different (50.0 %)
    [left]: Index(['a', 'b'], dtype='object')
    right]: Index(['a', 'c'], dtype='object')

    >>> df2 = cudf.DataFrame({"a":[1, 2], "b":[1.0, 3.0]}, index=[1, 2])
    >>> cudf.testing.assert_frame_equal(df1, df2)
    ......
    ......
    AssertionError: Column name="b" are different

    values are different (50.0 %)
    [left]:  [1. 2.]
    [right]: [1. 3.]

    # This will pass without any hitch
    >>> df2 = cudf.DataFrame({"a":[1, 2], "b":[1.0, 2.0]}, index=[1, 2])
    >>> cudf.testing.assert_frame_equal(df1, df2)
    """
    _check_isinstance(left, right, cudf.DataFrame)

    if check_frame_type:
        assert isinstance(left, type(right))

    # shape comparison
    if left.shape != right.shape:
        raise AssertionError("left and right shape mismatch")

    if check_like:
        left, right = left.reindex(index=right.index), right
        right = right[list(left._data.names)]

    if check_less_precise:
        raise NotImplementedError("check_less_precise is not yet supported")

    # index comparison
    assert_index_equal(
        left.index,
        right.index,
        exact=check_index_type,
        check_names=check_names,
        check_less_precise=check_less_precise,
        check_exact=check_exact,
        check_categorical=check_categorical,
        obj=f"{obj}.index",
    )

    pd.testing.assert_index_equal(
        left.columns,
        right.columns,
        exact=check_column_type,
        check_names=check_names,
        check_less_precise=check_less_precise,
        check_exact=check_exact,
        check_categorical=check_categorical,
        obj=f"{obj}.columns",
    )

    for col in left.columns:
        assert_column_equal(
            left._data[col],
            right._data[col],
            check_dtype=check_dtype,
            check_less_precise=check_less_precise,
            check_exact=check_exact,
            check_datetimelike_compat=check_datetimelike_compat,
            check_categorical=check_categorical,
            obj=f'Column name="{col}"',
        )
