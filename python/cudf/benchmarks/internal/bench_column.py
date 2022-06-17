import pytest
import pytest_cases
from utils import accepts_cudf_fixture, make_boolean_mask_column, make_gather_map


@accepts_cudf_fixture(cls="column", dtype="float")
def bench_apply_boolean_mask(benchmark, column):
    mask = make_boolean_mask_column(column.size)
    benchmark(column.apply_boolean_mask, mask)


@accepts_cudf_fixture(cls="column", dtype="float")
@pytest.mark.parametrize("dropnan", [True, False])
def bench_dropna(benchmark, column, dropnan):
    benchmark(column.dropna, drop_nan=dropnan)


@accepts_cudf_fixture(cls="column", dtype="float")
def bench_unique_single_column(benchmark, column):
    benchmark(column.unique)


@accepts_cudf_fixture(cls="column", dtype="float")
@pytest.mark.parametrize("nullify", [True, False])
@pytest.mark.parametrize("gather_how", ["sequence", "reverse", "random"])
def bench_take(benchmark, column, gather_how, nullify):
    gather_map = make_gather_map(column.size * 0.4, column.size, gather_how)._column
    benchmark(column.take, gather_map, nullify=nullify)


# TODO: Due to https://github.com/smarie/python-pytest-cases/issues/280 we
# cannot use the accepts_cudf_fixture decorator for cases. If and when that is
# resolved, we can change all of the cases below to use that instead of
# hardcoding the fixture name.
def setitem_case_stride_1_slice_scalar(column_dtype_int_nulls_false):
    return column_dtype_int_nulls_false, slice(None, None, 1), 42


def setitem_case_stride_2_slice_scalar(column_dtype_int_nulls_false):
    return column_dtype_int_nulls_false, slice(None, None, 2), 42


def setitem_case_boolean_column_scalar(column_dtype_int_nulls_false):
    column = column_dtype_int_nulls_false
    return column, [True, False] * (len(column) // 2), 42


def setitem_case_int_column_scalar(column_dtype_int_nulls_false):
    column = column_dtype_int_nulls_false
    return column, list(range(len(column))), 42


def setitem_case_stride_1_slice_align_to_key_size(column_dtype_int_nulls_false):
    column = column_dtype_int_nulls_false
    key = slice(None, None, 1)
    start, stop, stride = key.indices(len(column))
    materialized_key_size = len(column.slice(start, stop, stride))
    return column, key, [42] * materialized_key_size


def setitem_case_stride_2_slice_align_to_key_size(column_dtype_int_nulls_false):
    column = column_dtype_int_nulls_false
    key = slice(None, None, 2)
    start, stop, stride = key.indices(len(column))
    materialized_key_size = len(column.slice(start, stop, stride))
    return column, key, [42] * materialized_key_size


def setitem_case_boolean_column_align_to_col_size(column_dtype_int_nulls_false):
    column = column_dtype_int_nulls_false
    size = len(column)
    return column, [True, False] * (size // 2), [42] * size


def setitem_case_int_column_align_to_col_size(column_dtype_int_nulls_false):
    column = column_dtype_int_nulls_false
    size = len(column)
    return column, list(range(size)), [42] * size


# Benchmark Grid
# key:  slice == 1  (fill or copy_range shortcut),
#       slice != 1  (scatter),
#       column(bool)    (boolean_mask_scatter),
#       column(int) (scatter)
# value:    scalar,
#           column (len(val) == len(key)),
#           column (len(val) != len(key) & len == num_true)


@pytest_cases.parametrize_with_cases("column,key,value", cases=".", prefix="setitem")
def bench_setitem(benchmark, column, key, value):
    benchmark(column.__setitem__, key, value)
