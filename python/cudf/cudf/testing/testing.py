# Copyright (c) 2020-2022, NVIDIA CORPORATION.

from __future__ import annotations

from typing import Union

import cupy as cp
import numpy as np
import pandas as pd

import cudf
from cudf._lib.unary import is_nan
from cudf.api.types import (
    is_categorical_dtype,
    is_decimal_dtype,
    is_interval_dtype,
    is_list_dtype,
    is_numeric_dtype,
    is_string_dtype,
    is_struct_dtype,
)
from cudf.core._compat import PANDAS_GE_110
from cudf.core.missing import NA


def dtype_can_compare_equal_to_other(dtype):
    # return True if values of this dtype can compare
    # as equal to equal values of a different dtype
    return not (
        is_string_dtype(dtype)
        or is_list_dtype(dtype)
        or is_struct_dtype(dtype)
        or is_decimal_dtype(dtype)
        or is_interval_dtype(dtype)
    )


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
    rtol=1e-05,
    atol=1e-08,
    obj="ColumnBase",
):
    """
    Check that left and right columns are equal

    This function is intended to compare two columns and output
    any differences. Additional parameters allow varying the strictness
    of the equality checks performed.

    Parameters
    ----------
    left : Column
        left Column to compare
    right : Column
        right Column to compare
    check_dtype : bool, default True
        Whether to check the Column dtype is identical.
    check_column_type : bool or {'equiv'}, default 'equiv'
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
    rtol : float, default 1e-5
        Relative tolerance. Only used when `check_exact` is False.
    atol : float, default 1e-8
        Absolute tolerance. Only used when `check_exact` is False.
    obj : str, default 'ColumnBase'
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
    else:
        if left.null_count == len(left) and right.null_count == len(right):
            return True

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
            left_cat = left.categories
            right_cat = right.categories

            if check_category_order:
                assert_index_equal(
                    left_cat,
                    right_cat,
                    exact=check_dtype,
                    check_exact=True,
                    check_categorical=False,
                    rtol=rtol,
                    atol=atol,
                )
                assert_column_equal(
                    left.codes,
                    right.codes,
                    check_dtype=check_dtype,
                    check_exact=True,
                    check_categorical=False,
                    check_category_order=False,
                    rtol=rtol,
                    atol=atol,
                )

            if left.ordered != right.ordered:
                msg1 = f"{left.ordered}"
                msg2 = f"{right.ordered}"
                raise_assert_detail(
                    f"{obj} category", "Orders are different", msg1, msg2
                )

    if (
        not check_dtype
        and is_categorical_dtype(left)
        and is_categorical_dtype(right)
    ):
        left = left.astype(left.categories.dtype)
        right = right.astype(right.categories.dtype)
    columns_equal = False
    if left.size == right.size == 0:
        columns_equal = True
    elif not (
        (
            not dtype_can_compare_equal_to_other(left.dtype)
            and is_numeric_dtype(right)
        )
        or (
            is_numeric_dtype(left)
            and not dtype_can_compare_equal_to_other(right)
        )
    ):
        try:
            # nulls must be in the same places for all dtypes
            columns_equal = cp.all(
                left.isnull().values == right.isnull().values
            )

            if columns_equal and not check_exact and is_numeric_dtype(left):
                # non-null values must be the same
                columns_equal = cp.allclose(
                    left.apply_boolean_mask(
                        left.isnull().unary_operator("not")
                    ).values,
                    right.apply_boolean_mask(
                        right.isnull().unary_operator("not")
                    ).values,
                )
                if columns_equal and (
                    left.dtype.kind == right.dtype.kind == "f"
                ):
                    columns_equal = cp.all(
                        is_nan(left).values == is_nan(right).values
                    )
            else:
                columns_equal = left.equals(right)
        except TypeError as e:
            if str(e) != "Categoricals can only compare with the same type":
                raise e
            else:
                columns_equal = False
            if is_categorical_dtype(left) and is_categorical_dtype(right):
                left = left.astype(left.categories.dtype)
                right = right.astype(right.categories.dtype)
    if not columns_equal:
        ldata = str([val for val in left.to_pandas(nullable=True)])
        rdata = str([val for val in right.to_pandas(nullable=True)])
        try:
            diff = 0
            for i in range(left.size):
                if not null_safe_scalar_equals(left[i], right[i]):
                    diff += 1
            diff = diff * 100.0 / left.size
        except BaseException:
            diff = 100.0
        raise_assert_detail(
            obj,
            f"values are different ({np.round(diff, 5)} %)",
            {ldata},
            {rdata},
        )


def null_safe_scalar_equals(left, right):
    if left in {NA, np.nan} or right in {NA, np.nan}:
        return left is right
    return left == right


def assert_index_equal(
    left,
    right,
    exact="equiv",
    check_names: bool = True,
    check_less_precise: Union[bool, int] = False,
    check_exact: bool = True,
    check_categorical: bool = True,
    check_order: bool = True,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    obj: str = "Index",
):
    """
    Check that left and right Index are equal

    This function is intended to compare two Index and output
    any differences. Additional parameters allow varying the strictness
    of the equality checks performed.

    Parameters
    ----------
    left : Index
        left Index to compare
    right : Index
        right Index to compare
    exact : bool or {'equiv'}, default 'equiv'
        Whether to check the Index class, dtype and inferred_type
        are identical. If 'equiv', then RangeIndex can be substituted
        for Int8Index, Int16Index, Int32Index, Int64Index as well.
    check_names : bool, default True
        Whether to check the names attribute.
    check_less_precise : bool or int, default False
        Not yet supported
    check_exact : bool, default False
        Whether to compare number exactly.
    check_categorical : bool, default True
        Whether to compare internal Categorical exactly.
    check_order : bool, default True
        Whether to compare the order of index entries as
        well as their values.
        If True, both indexes must contain the same elements,
        in the same order.
        If False, both indexes must contain the same elements,
        but in any order.
    rtol : float, default 1e-5
        Relative tolerance. Only used when `check_exact` is False.
    atol : float, default 1e-8
        Absolute tolerance. Only used when `check_exact` is False.
    obj : str, default 'Index'
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
    <BLANKLINE>
    values are different (25.0 %)
    [left]:  [1 2 3 4]
    [right]: [1 2 3 5]

    >>> id2 = cudf.Index([1, 2, 3, 4], name="b")
    >>> cudf.testing.assert_index_equal(id1, id2)
    ......
    ......
    AssertionError: Index are different
    <BLANKLINE>
    name mismatch
    [left]:  a
    [right]: b

    This will pass without any hitch:

    >>> id2 = cudf.Index([1, 2, 3, 4], name="a")
    >>> cudf.testing.assert_index_equal(id1, id2)
    """

    # instance validation
    _check_isinstance(left, right, cudf.BaseIndex)

    _check_types(
        left, right, exact=exact, check_categorical=check_categorical, obj=obj
    )

    if len(left) != len(right):
        raise_assert_detail(
            obj, "lengths are different", f"{len(left)}", f"{len(right)}"
        )

    # If order doesn't matter then sort the index entries
    if not check_order:
        left = left.sort_values()
        right = right.sort_values()

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
                check_exact=check_exact,
                check_less_precise=check_less_precise,
                check_order=check_order,
                rtol=rtol,
                atol=atol,
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
    rtol=1e-5,
    atol=1e-8,
    obj="Series",
):
    """
    Check that left and right Series are equal

    This function is intended to compare two Series and output
    any differences. Additional parameters allow varying the strictness
    of the equality checks performed.

    Parameters
    ----------
    left : Series
        left Series to compare
    right : Series
        right Series to compare
    check_dtype : bool, default True
        Whether to check the Series dtype is identical.
    check_index_type : bool or {'equiv'}, default 'equiv'
        Whether to check the Index class, dtype and inferred_type
        are identical.
    check_series_type : bool, default True
        Whether to check the series class, dtype and
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
    rtol : float, default 1e-5
        Relative tolerance. Only used when `check_exact` is False.
    atol : float, default 1e-8
        Absolute tolerance. Only used when `check_exact` is False.
    obj : str, default 'Series'
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
    <BLANKLINE>
    values are different (25.0 %)
    [left]:  [1 2 3 4]
    [right]: [1 2 3 5]

    >>> sr2 = cudf.Series([1, 2, 3, 4], name="b")
    >>> cudf.testing.assert_series_equal(sr1, sr2)
    ......
    ......
    AssertionError: Series are different
    <BLANKLINE>
    name mismatch
    [left]:  a
    [right]: b

    This will pass without any hitch:

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
        rtol=rtol,
        atol=atol,
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
        rtol=rtol,
        atol=atol,
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
    check_names=True,
    by_blocks=False,
    check_exact=False,
    check_datetimelike_compat=False,
    check_categorical=True,
    check_like=False,
    rtol=1e-5,
    atol=1e-8,
    obj="DataFrame",
):
    """
    Check that left and right DataFrame are equal

    This function is intended to compare two DataFrame and output
    any differences. Additional parameters allow varying the strictness
    of the equality checks performed.

    Parameters
    ----------
    left : DataFrame
        left DataFrame to compare
    right : DataFrame
        right DataFrame to compare
    check_dtype : bool, default True
        Whether to check the DataFrame dtype is identical.
    check_index_type : bool or {'equiv'}, default 'equiv'
        Whether to check the Index class, dtype and inferred_type
        are identical.
    check_column_type : bool, default True
        Whether to check the column class, dtype and
        inferred_type are identical. Currently it is idle,
        and similar to pandas.
    check_frame_type : bool, default True
        Whether to check the DataFrame class is identical.
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
    rtol : float, default 1e-5
        Relative tolerance. Only used when `check_exact` is False.
    atol : float, default 1e-8
        Absolute tolerance. Only used when `check_exact` is False.
    obj : str, default 'DataFrame'
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
    <BLANKLINE>
    values are different (100.0 %)
    [left]:  [1 2]
    [right]: [2 3]

    >>> df2 = cudf.DataFrame({"a":[1, 2], "c":[1.0, 2.0]}, index=[1, 2])
    >>> cudf.testing.assert_frame_equal(df1, df2)
    ......
    ......
    AssertionError: DataFrame.columns are different
    <BLANKLINE>
    DataFrame.columns values are different (50.0 %)
    [left]: Index(['a', 'b'], dtype='object')
    right]: Index(['a', 'c'], dtype='object')

    >>> df2 = cudf.DataFrame({"a":[1, 2], "b":[1.0, 3.0]}, index=[1, 2])
    >>> cudf.testing.assert_frame_equal(df1, df2)
    ......
    ......
    AssertionError: Column name="b" are different
    <BLANKLINE>
    values are different (50.0 %)
    [left]:  [1. 2.]
    [right]: [1. 3.]

    This will pass without any hitch:

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

    # index comparison
    assert_index_equal(
        left.index,
        right.index,
        exact=check_index_type,
        check_names=check_names,
        check_exact=check_exact,
        check_categorical=check_categorical,
        rtol=rtol,
        atol=atol,
        obj=f"{obj}.index",
    )

    if PANDAS_GE_110:
        pd.testing.assert_index_equal(
            left._data.to_pandas_index(),
            right._data.to_pandas_index(),
            exact=check_column_type,
            check_names=check_names,
            check_exact=check_exact,
            check_categorical=check_categorical,
            rtol=rtol,
            atol=atol,
            obj=f"{obj}.columns",
        )
    else:
        pd.testing.assert_index_equal(
            left._data.to_pandas_index(),
            right._data.to_pandas_index(),
            exact=check_column_type,
            check_names=check_names,
            check_exact=check_exact,
            check_categorical=check_categorical,
            obj=f"{obj}.columns",
        )

    for col in left._column_names:
        assert_column_equal(
            left._data[col],
            right._data[col],
            check_dtype=check_dtype,
            check_exact=check_exact,
            check_datetimelike_compat=check_datetimelike_compat,
            check_categorical=check_categorical,
            rtol=rtol,
            atol=atol,
            obj=f'Column name="{col}"',
        )
