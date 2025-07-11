# Copyright (c) 2025, NVIDIA CORPORATION.

import itertools

import pyarrow as pa
import pytest
from utils import assert_column_eq

import pylibcudf as plc


@pytest.fixture
def children_arrow():
    return [pa.array([1, None, 2]), pa.array([10, None, 20])]


@pytest.fixture
def children_plc(children_arrow):
    return [plc.Column.from_arrow(child) for child in children_arrow]


def test_as_struct_column_empty_children():
    with pytest.raises(ValueError):
        plc.Column.as_struct_column([])


def test_as_struct_column_children_not_column():
    with pytest.raises(ValueError):
        plc.Column.as_struct_column([[1, 2, 3], [4, 5, 6]])


def test_as_struct_column_children_different_sizes(children_plc):
    new_child_size = children_plc[0].size() + 1
    children = itertools.chain(
        children_plc, [plc.Column.from_arrow(pa.array([1] * new_child_size))]
    )
    with pytest.raises(ValueError):
        plc.Column.as_struct_column(children)


def test_as_struct_column_struct_list(children_plc):
    result = plc.Column.as_struct_column(children_plc)
    expected = plc.interop.to_arrow(
        plc.Column(
            plc.DataType(plc.TypeId.STRUCT),
            children_plc[0].size(),
            None,
            children_plc[0].null_mask(),
            children_plc[0].null_count(),
            0,
            children_plc,
        )
    )
    assert_column_eq(result, expected)
