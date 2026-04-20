# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import operator

import pandas as pd
import pytest
from pandas.errors import DuplicateLabelError

import cudf
from cudf.core.indexed_frame import Flags
from cudf.testing import assert_eq


@pytest.fixture(params=[cudf.Series, cudf.DataFrame])
def frame_or_series(request):
    return request.param


def _make(cls, data=None):
    if data is None:
        data = [1, 2, 3]
    if cls is cudf.Series:
        return cls(data)
    return cls({"a": data})


def test_flags_default_allows_duplicates(frame_or_series):
    obj = _make(frame_or_series)
    assert obj.flags.allows_duplicate_labels is True


def test_flags_setter_attribute(frame_or_series):
    obj = _make(frame_or_series)
    obj.flags.allows_duplicate_labels = False
    assert obj.flags.allows_duplicate_labels is False


def test_flags_getitem_and_setitem(frame_or_series):
    obj = _make(frame_or_series)
    assert obj.flags["allows_duplicate_labels"] is True
    obj.flags["allows_duplicate_labels"] = False
    assert obj.flags["allows_duplicate_labels"] is False


def test_flags_unknown_key_raises(frame_or_series):
    obj = _make(frame_or_series)
    with pytest.raises(KeyError):
        obj.flags["nonexistent"]
    with pytest.raises(ValueError):
        obj.flags["nonexistent"] = True


def test_flags_repr(frame_or_series):
    obj = _make(frame_or_series)
    assert repr(obj.flags) == "<Flags(allows_duplicate_labels=True)>"
    obj.flags.allows_duplicate_labels = False
    assert repr(obj.flags) == "<Flags(allows_duplicate_labels=False)>"


def test_flags_equality(frame_or_series):
    a = _make(frame_or_series)
    b = _make(frame_or_series)
    assert a.flags == b.flags
    b.flags.allows_duplicate_labels = False
    assert a.flags != b.flags
    assert a.flags != object()


def test_flags_setter_casts_to_bool(frame_or_series):
    obj = _make(frame_or_series)
    obj.flags.allows_duplicate_labels = 0
    assert obj.flags.allows_duplicate_labels is False
    obj.flags.allows_duplicate_labels = 1
    assert obj.flags.allows_duplicate_labels is True


def test_flags_rejects_duplicates_when_disabled():
    df = cudf.DataFrame({"a": [1, 2, 3]}, index=[1, 1, 2])
    with pytest.raises(DuplicateLabelError):
        df.flags.allows_duplicate_labels = False


def test_flags_allows_non_unique_when_enabled():
    df = cudf.DataFrame({"a": [1, 2]}, index=[1, 1])
    df.flags.allows_duplicate_labels = True
    assert df.flags.allows_duplicate_labels is True


def test_flags_weakref_dead_obj_raises():
    # Exercise the ValueError branch when the weakref target is gone.
    df = cudf.DataFrame({"a": [1]})
    flags = df.flags
    del df
    with pytest.raises(ValueError, match="deleted"):
        flags.allows_duplicate_labels = False


def test_set_flags_returns_new_object(frame_or_series):
    obj = _make(frame_or_series)
    new = obj.set_flags(allows_duplicate_labels=False)
    assert new is not obj
    assert obj.flags.allows_duplicate_labels is True
    assert new.flags.allows_duplicate_labels is False


def test_set_flags_no_change_when_none(frame_or_series):
    obj = _make(frame_or_series)
    obj.flags.allows_duplicate_labels = False
    new = obj.set_flags()
    assert new.flags.allows_duplicate_labels is False


def test_set_flags_copy_true_makes_deep_copy():
    df = cudf.DataFrame({"a": [1, 2, 3]})
    new = df.set_flags(copy=True, allows_duplicate_labels=False)
    new["a"][0] = 99
    # Original left untouched.
    assert df["a"][0] == 1


def test_copy_preserves_flags(frame_or_series):
    obj = _make(frame_or_series).set_flags(allows_duplicate_labels=False)
    assert obj.copy().flags.allows_duplicate_labels is False
    assert obj.copy(deep=False).flags.allows_duplicate_labels is False


@pytest.mark.parametrize(
    "left_flag, right_flag, expected",
    [
        (True, True, True),
        (True, False, False),
        (False, True, False),
        (False, False, False),
    ],
)
def test_binop_flags_anded(frame_or_series, left_flag, right_flag, expected):
    left = _make(frame_or_series).set_flags(allows_duplicate_labels=left_flag)
    right = _make(frame_or_series).set_flags(
        allows_duplicate_labels=right_flag
    )
    assert (left + right).flags.allows_duplicate_labels is expected


def test_binop_with_scalar_preserves_flags(frame_or_series):
    obj = _make(frame_or_series).set_flags(allows_duplicate_labels=False)
    assert (obj + 1).flags.allows_duplicate_labels is False
    assert (1 + obj).flags.allows_duplicate_labels is False


def test_binop_attrs_right_overrides_left():
    a = cudf.Series([1])
    b = cudf.Series([1])
    a.attrs = {"a": 1}
    b.attrs = {"b": 2}
    # Mirrors pandas' __finalize__(self).__finalize__(other): right wins
    # when both sides carry attrs.
    assert (a + b).attrs == {"b": 2}


def test_binop_attrs_inherit_from_non_empty_side():
    a = cudf.Series([1])
    b = cudf.Series([1])
    b.attrs = {"x": 1}
    assert (a + b).attrs == {"x": 1}
    a.attrs = {"y": 2}
    b.attrs = {}
    assert (a + b).attrs == {"y": 2}


@pytest.mark.parametrize(
    "left_flag, right_flag, expected",
    [
        (True, True, True),
        (True, False, False),
        (False, True, False),
        (False, False, False),
    ],
)
@pytest.mark.parametrize("how", ["left", "right", "inner", "outer", "cross"])
def test_merge_flags_anded(how, left_flag, right_flag, expected):
    left = cudf.DataFrame({"test": [1]}).set_flags(
        allows_duplicate_labels=left_flag
    )
    right = cudf.DataFrame({"test": [1]}).set_flags(
        allows_duplicate_labels=right_flag
    )
    if how == "cross":
        result = left.merge(right, how=how)
    else:
        result = left.merge(right, how=how, on="test")
    assert result.flags.allows_duplicate_labels is expected


def test_merge_propagates_attrs_only_when_equal():
    left = cudf.DataFrame({"test": [1]})
    right = cudf.DataFrame({"test": [1]})
    left.attrs = {"a": [1, 2]}
    right.attrs = {"a": [1, 2]}
    assert left.merge(right, how="inner", on="test").attrs == {"a": [1, 2]}


def test_merge_drops_attrs_when_inputs_differ():
    left = cudf.DataFrame({"test": [1]})
    right = cudf.DataFrame({"test": [1]})
    left.attrs = {"a": 1}
    right.attrs = {"b": 2}
    assert left.merge(right, how="inner", on="test").attrs == {}


def test_merge_drops_attrs_when_one_side_empty():
    left = cudf.DataFrame({"test": [1]})
    right = cudf.DataFrame({"test": [1]})
    left.attrs = {"a": 1}
    # right has no attrs
    assert left.merge(right, how="inner", on="test").attrs == {}


def test_from_pandas_preserves_flags(frame_or_series):
    if frame_or_series is cudf.Series:
        src = pd.Series([1, 2]).set_flags(allows_duplicate_labels=False)
    else:
        src = pd.DataFrame({"a": [1, 2]}).set_flags(
            allows_duplicate_labels=False
        )
    out = cudf.from_pandas(src)
    assert out.flags.allows_duplicate_labels is False


def test_to_pandas_preserves_flags(frame_or_series):
    obj = _make(frame_or_series).set_flags(allows_duplicate_labels=False)
    assert obj.to_pandas().flags.allows_duplicate_labels is False


def test_roundtrip_through_pandas_preserves_flags(frame_or_series):
    obj = _make(frame_or_series).set_flags(allows_duplicate_labels=False)
    assert (
        cudf.from_pandas(obj.to_pandas()).flags.allows_duplicate_labels
        is False
    )


def test_dataframe_constructor_preserves_flags_from_pandas():
    pdf = pd.DataFrame({"a": [1, 2]}).set_flags(allows_duplicate_labels=False)
    assert cudf.DataFrame(pdf).flags.allows_duplicate_labels is False


def test_series_constructor_preserves_flags_from_pandas():
    ps = pd.Series([1, 2]).set_flags(allows_duplicate_labels=False)
    assert cudf.Series(ps).flags.allows_duplicate_labels is False


def test_dataframe_constructor_preserves_flags_from_cudf():
    src = cudf.DataFrame({"a": [1, 2]}).set_flags(
        allows_duplicate_labels=False
    )
    assert cudf.DataFrame(src).flags.allows_duplicate_labels is False


def test_series_constructor_preserves_flags_from_cudf():
    src = cudf.Series([1, 2]).set_flags(allows_duplicate_labels=False)
    assert cudf.Series(src).flags.allows_duplicate_labels is False


def test_from_data_accepts_allows_duplicate_labels():
    df = cudf.DataFrame._from_data(
        cudf.DataFrame({"a": [1]})._data,
        allows_duplicate_labels=False,
    )
    assert df.flags.allows_duplicate_labels is False


def test_matches_pandas_behavior_set_flags(frame_or_series):
    cu = _make(frame_or_series).set_flags(allows_duplicate_labels=False)
    pd_cls = pd.Series if frame_or_series is cudf.Series else pd.DataFrame
    pd_obj = (
        pd_cls([1, 2, 3])
        if frame_or_series is cudf.Series
        else pd_cls({"a": [1, 2, 3]})
    ).set_flags(allows_duplicate_labels=False)
    assert (
        cu.flags.allows_duplicate_labels
        == pd_obj.flags.allows_duplicate_labels
    )
    assert_eq(cu, pd_obj)


def test_flags_class_is_exposed():
    # Used by users who want to type-hint or isinstance-check.
    assert isinstance(cudf.DataFrame().flags, Flags)


def test_binop_attrs_deepcopied():
    a = cudf.Series([1])
    b = cudf.Series([2])
    b.attrs = {"nested": [1, 2]}
    result = a + b
    # Mutating the result's attrs must not affect the source.
    result.attrs["nested"].append(3)
    assert b.attrs["nested"] == [1, 2]


def test_binop_flag_propagation_with_operators():
    a = cudf.Series([1, 2]).set_flags(allows_duplicate_labels=False)
    b = cudf.Series([3, 4])
    for op in (
        operator.add,
        operator.sub,
        operator.mul,
        operator.truediv,
        operator.floordiv,
    ):
        assert op(a, b).flags.allows_duplicate_labels is False
