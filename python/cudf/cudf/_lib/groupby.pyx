# Copyright (c) 2020-2024, NVIDIA CORPORATION.
from functools import singledispatch

from pandas.errors import DataError

from cudf.api.types import _is_categorical_dtype, is_string_dtype
from cudf.core.buffer import acquire_spill_lock
from cudf.core.dtypes import (
    CategoricalDtype,
    DecimalDtype,
    IntervalDtype,
    ListDtype,
    StructDtype,
)

from cudf._lib.scalar cimport DeviceScalar
from cudf._lib.utils cimport columns_from_pylibcudf_table

from cudf._lib.scalar import as_device_scalar

from cudf._lib.pylibcudf.libcudf.replace cimport replace_policy
from cudf._lib.pylibcudf.libcudf.scalar.scalar cimport scalar

from cudf._lib import pylibcudf
from cudf._lib.aggregation import make_aggregation

# The sets below define the possible aggregations that can be performed on
# different dtypes. These strings must be elements of the AggregationKind enum.
# The libcudf infrastructure exists for "COLLECT" support on
# categoricals, but the dtype support in python does not.
_CATEGORICAL_AGGS = {"COUNT", "NUNIQUE", "SIZE", "UNIQUE"}
_STRING_AGGS = {
    "COLLECT",
    "COUNT",
    "MAX",
    "MIN",
    "NTH",
    "NUNIQUE",
    "SIZE",
    "UNIQUE",
}
_LIST_AGGS = {"COLLECT"}
_STRUCT_AGGS = {"COLLECT", "CORRELATION", "COVARIANCE"}
_INTERVAL_AGGS = {"COLLECT"}
_DECIMAL_AGGS = {
    "ARGMIN",
    "ARGMAX",
    "COLLECT",
    "COUNT",
    "MAX",
    "MIN",
    "NTH",
    "NUNIQUE",
    "SUM",
}
# workaround for https://github.com/cython/cython/issues/3885
ctypedef const scalar constscalar


@singledispatch
def get_valid_aggregation(dtype):
    if is_string_dtype(dtype):
        return _STRING_AGGS
    return "ALL"


@get_valid_aggregation.register
def _(dtype: ListDtype):
    return _LIST_AGGS


@get_valid_aggregation.register
def _(dtype: CategoricalDtype):
    return _CATEGORICAL_AGGS


@get_valid_aggregation.register
def _(dtype: ListDtype):
    return _LIST_AGGS


@get_valid_aggregation.register
def _(dtype: StructDtype):
    return _STRUCT_AGGS


@get_valid_aggregation.register
def _(dtype: IntervalDtype):
    return _INTERVAL_AGGS


@get_valid_aggregation.register
def _(dtype: DecimalDtype):
    return _DECIMAL_AGGS


cdef class GroupBy:
    cdef dict __dict__

    def __init__(self, keys, dropna=True):
        with acquire_spill_lock() as spill_lock:
            self._groupby = pylibcudf.groupby.GroupBy(
                pylibcudf.table.Table([c.to_pylibcudf(mode="read") for c in keys]),
                pylibcudf.types.NullPolicy.EXCLUDE if dropna
                else pylibcudf.types.NullPolicy.INCLUDE
            )

            # We spill lock the columns while this GroupBy instance is alive.
            self._spill_lock = spill_lock

    def groups(self, list values):
        """
        Perform a sort groupby, using the keys used to construct the Groupby as the key
        columns and ``values`` as the value columns.

        Parameters
        ----------
        values: list of Columns
            The value columns

        Returns
        -------
        offsets: list of integers
            Integer offsets such that offsets[i+1] - offsets[i]
            represents the size of group `i`.
        grouped_keys: list of Columns
            The grouped key columns
        grouped_values: list of Columns
            The grouped value columns
        """
        offsets, grouped_keys, grouped_values = self._groupby.get_groups(
            pylibcudf.table.Table([c.to_pylibcudf(mode="read") for c in values])
            if values else None
        )

        return (
            offsets,
            columns_from_pylibcudf_table(grouped_keys),
            (
                columns_from_pylibcudf_table(grouped_values)
                if grouped_values is not None else []
            ),
        )

    def aggregate(self, values, aggregations):
        """
        Parameters
        ----------
        values : Frame
        aggregations
            A dict mapping column names in `Frame` to a list of aggregations
            to perform on that column

            Each aggregation may be specified as:
            - a string (e.g., "max")
            - a lambda/function

        Returns
        -------
        Frame of aggregated values
        """
        included_aggregations = []
        column_included = []
        requests = []
        for i, (col, aggs) in enumerate(zip(values, aggregations)):
            valid_aggregations = get_valid_aggregation(col.dtype)
            included_aggregations_i = []
            col_aggregations = []
            for agg in aggs:
                str_agg = str(agg)
                if (
                    is_string_dtype(col)
                    and agg not in _STRING_AGGS
                    and
                    (
                        str_agg in {"cumsum", "cummin", "cummax"}
                        or not (
                        any(a in str_agg for a in {
                            "count",
                            "max",
                            "min",
                            "first",
                            "last",
                            "nunique",
                            "unique",
                            "nth"
                        })
                        or (agg is list)
                        )
                    )
                ):
                    raise TypeError(
                        f"function is not supported for this dtype: {agg}"
                    )
                elif (
                    _is_categorical_dtype(col)
                    and agg not in _CATEGORICAL_AGGS
                    and (
                        str_agg in {"cumsum", "cummin", "cummax"}
                        or
                        not (
                            any(a in str_agg for a in {"count", "max", "min", "unique"})
                        )
                    )
                ):
                    raise TypeError(
                        f"{col.dtype} type does not support {agg} operations"
                    )

                agg_obj = make_aggregation(agg)
                if valid_aggregations == "ALL" or agg_obj.kind in valid_aggregations:
                    included_aggregations_i.append((agg, agg_obj.kind))
                    col_aggregations.append(agg_obj.c_obj)
            included_aggregations.append(included_aggregations_i)
            if col_aggregations:
                requests.append(pylibcudf.groupby.GroupByRequest(
                    col.to_pylibcudf(mode="read"), col_aggregations
                ))
                column_included.append(i)

        if not requests and any(len(v) > 0 for v in aggregations):
            raise DataError("All requested aggregations are unsupported.")

        keys, results = self._groupby.scan(requests) if \
            _is_all_scan_aggregate(aggregations) else self._groupby.aggregate(requests)

        result_columns = [[] for _ in range(len(values))]
        for i, result in zip(column_included, results):
            result_columns[i] = columns_from_pylibcudf_table(result)

        return result_columns, columns_from_pylibcudf_table(keys), included_aggregations

    def shift(self, list values, int periods, list fill_values):
        keys, shifts = self._groupby.shift(
            pylibcudf.table.Table([c.to_pylibcudf(mode="read") for c in values]),
            [periods] * len(values),
            [
                (<DeviceScalar> as_device_scalar(val, dtype=col.dtype)).c_value
                for val, col in zip(fill_values, values)
            ],
        )

        return columns_from_pylibcudf_table(shifts), columns_from_pylibcudf_table(keys)

    def replace_nulls(self, list values, object method):
        # TODO: This is using an enum (replace_policy) that has not been exposed in
        # pylibcudf yet. We'll want to fix that import once it is in pylibcudf.
        _, replaced = self._groupby.replace_nulls(
            pylibcudf.table.Table([c.to_pylibcudf(mode="read") for c in values]),
            [
                replace_policy.PRECEDING
                if method == 'ffill' else replace_policy.FOLLOWING
            ] * len(values),
        )

        return columns_from_pylibcudf_table(replaced)


_GROUPBY_SCANS = {"cumcount", "cumsum", "cummin", "cummax", "cumprod", "rank"}


def _is_all_scan_aggregate(all_aggs):
    """
    Returns true if all are scan aggregations.
    Raises
    ------
    NotImplementedError
        If both reduction aggregations and scan aggregations are present.
    """

    def get_name(agg):
        return agg.__name__ if callable(agg) else agg

    all_scan = all(
        get_name(agg_name) in _GROUPBY_SCANS for aggs in all_aggs
        for agg_name in aggs
    )
    any_scan = any(
        get_name(agg_name) in _GROUPBY_SCANS for aggs in all_aggs
        for agg_name in aggs
    )

    if not all_scan and any_scan:
        raise NotImplementedError(
            "Cannot perform both aggregation and scan in one operation"
        )
    return all_scan and any_scan
