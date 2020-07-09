# Copyright (c) 2020, NVIDIA CORPORATION.

import cudf
from cudf.core.column.categorical import CategoricalColumn
from typing import Union
from cudf.utils.dtypes import is_categorical_dtype
import pandas as pd
import numpy as np

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


def _check_types(l, r, check_categorical=True, exact="equiv", obj="Index"):
    if exact != True:
        if (isinstance(l, cudf.RangeIndex) and isinstance(r, cudf.Int64Index)) or (isinstance(r, cudf.RangeIndex) and isinstance(l, cudf.Int64Index)):
            return 

    if type(l) != type(r):
        raise AssertionError(f"{obj} left and right type differ, left is of type {type(l)} and right is of type {type(r)}")
    

    if exact and is_categorical_dtype(l):
        #if not l.equals(r):
        if l.dtype != r.dtype:
            raise AssertionError(f"{obj} has Catgorical difference between left {l} and right {r}")


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
    obj="ColumnBase"
):
    if check_dtype:
        if (
            is_categorical_dtype(left)
            and is_categorical_dtype(right)
            and not check_categorical
        ):
            pass
        else:
            if left.dtype != right.dtype:
                msg1 = f"{left.dtype}"
                msg2 = f"{right.dtype}"
                raise_assert_detail(obj, "Dtypes are different", msg1, msg2)

    if check_datetimelike_compat:
        if not cudf.Index(left).equals(cudf.Index(right)):
            raise AssertionError(f"[datetimelike_compat=True] {left.values} "
                f"is not equal to {right.values}."
            )

        return 

    if check_exact and check_categorical:
        if is_categorical_dtype(left) and is_categorical_dtype(right):
            left_cat = left.cat().categories
            right_cat = right.cat().categories

            if check_category_order:
                assert_index_equal(left_cat, right_cat, exact=check_dtype, check_exact=True, check_categorical=False)
                assert_column_equal(left.codes, right.codes, check_dtype=check_dtype, check_exact=True, check_categorical=False, check_category_order=False)

            if left.ordered != right.ordered:
                msg1 = f"{left.ordered}"
                msg2 = f"{right.ordered}"
                raise_assert_detail("{obj} category", "Orders are different", msg1, msg2)

    if not check_dtype and is_categorical_dtype(left) and is_categorical_dtype(right):
        tmp_left = left.astype(left.categories.dtype)
        tmp_right = right.astype(right.categories.dtype)
        if not tmp_left.equals(tmp_right):
            msg1 = f"{left}"
            msg2 = f"{right}"
            diff = tmp_left.apply_boolean_mask(tmp_left.binary_operator("ne", tmp_right)).size
            diff = diff * 100.0/left.size
            raise_assert_detail(obj, f"values are different ({np.round(diff, 5)} %)", msg1, msg2)

    else:
        columns_equal = False
        try:
            columns_equal = left.equals(right)
        except TypeError:
            if is_categorical_dtype(left) and is_categorical_dtype(right):
                left = left.astype(left.categories.dtype)
                right = right.astype(right.categories.dtype)
        if not columns_equal:
            msg1 = f"{left.to_array()}"
            msg2 = f"{right.to_array()}"
            diff = left.apply_boolean_mask(left.binary_operator("ne", right)).size
            diff = diff * 100.0/left.size
            raise_assert_detail(obj, f"values are different ({np.round(diff, 5)} %)", msg1, msg2)


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

    # instance validation
    _check_isinstance(left, right, cudf.Index)

    _check_types(left, right, exact=exact, check_categorical=check_categorical, obj=obj)

    if len(left) != len(right):
        raise_assert_detail(obj, "lengths are different", f"{len(left)}", f"{len(right)}")

    if isinstance(left, cudf.MultiIndex):
        if left.nlevels != right.nlevels:
            raise AssertionError("Number of levels mismatch, left has {l.nlevels} levels and right has {r.nlevels}")

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

    assert_column_equal(left._columns[0], right._columns[0], check_dtype=exact, check_exact=check_exact,
                            check_categorical=check_categorical)

    # metadata comparison
    if check_names and (left.name != right.name):
        raise_assert_detail(obj, "name mismatch", "{left.name}", "{right.name}")


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

    assert_column_equal(left._column, right._column, check_dtype=check_dtype, check_column_type=check_series_type, check_less_precise=check_less_precise, check_exact=check_exact, check_datetimelike_compat=check_datetimelike_compat, check_categorical=check_categorical, check_category_order=check_category_order)

    # metadata comparison
    if check_names and (left.name != right.name):
        raise_assert_detail(obj, "name mismatch", "{left.name}", "{right.name}")


def assert_frame_equal(
    left,
    right,
    check_dtype=True,
    check_index_type="equiv",
    check_column_type="equiv",
    check_frame_type=True,
    check_less_precise=False,
    by_blocks = False,
    check_names=True,
    check_exact=False,
    check_datetimelike_compat=False,
    check_categorical=True,
    check_like=False,
    obj="DataFrame",
):

    _check_isinstance(left, right, cudf.DataFrame)

    if check_frame_type:
        assert isinstance(left, type(right))

    # shape comparison
    if left.shape != right.shape:
        raise AssertionError("left and right shape mismatch")

    if check_like:
        left, right = left.reindex(index=right.index), right
        right = right[list(left._data.names)]

    if check_less_precise != False:
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
        exact=check_index_type,
        check_names=check_names,
        check_less_precise=check_less_precise,
        check_exact=check_exact,
        check_categorical=check_categorical,
        obj=f"{obj}.columns",
    )

    if by_blocks: 
        raise NotImplementedError("by_blocks is not supported")
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


