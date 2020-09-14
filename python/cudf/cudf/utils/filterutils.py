# Copyright (c) 2020, NVIDIA CORPORATION.


from numba import cuda
import numpy as np
import cupy
import datetime

import pandas as pd
import cudf
import dask.dataframe as dd
import dask.array as da


def _apply_filter_bool_eq(val, col_stats):
    if "true_count" in col_stats and "false_count" in col_stats:
        if val is True:
            if (col_stats["true_count"] == 0) or (
                col_stats["false_count"] == col_stats["number_of_values"]
            ):
                return False
        elif val is False:
            if (col_stats["false_count"] == 0) or (
                col_stats["true_count"] == col_stats["number_of_values"]
            ):
                return False
    return True


def _apply_filter_not_eq(val, col_stats):
    return ("minimum" in col_stats and val < col_stats["minimum"]) or (
        "maximum" in col_stats and val > col_stats["maximum"]
    )


def _apply_predicate(op, val, col_stats):
    # Sanitize operator
    if op not in {"=", "==", "!=", "<", "<=", ">", ">=", "in", "not in"}:
        raise ValueError(
            '"{0}" is not a valid operator in predicates.'.format(op)
        )

    has_min = "minimum" in col_stats
    has_max = "maximum" in col_stats
    has_sum = "sum" in col_stats
    col_min = col_stats["minimum"] if has_min else None
    col_max = col_stats["maximum"] if has_max else None
    col_sum = col_stats["sum"] if has_sum else None

    # Apply operator
    if op == "=" or op == "==":
        if _apply_filter_not_eq(val, col_stats):
            return False
        if pd.isnull(val) and not col_stats["has_null"]:
            return False
        if not _apply_filter_bool_eq(val, col_stats):
            return False
    elif op == "!=":
        if has_min and has_max and val == col_min and val == col_max:
            return False
        if _apply_filter_bool_eq(val, col_stats):
            return False
    elif has_min and (
        (op == "<" and val <= col_min) or (op == "<=" and val < col_min)
    ):
        return False
    elif has_max and (
        (op == ">" and val >= col_max) or (op == ">=" and val > col_max)
    ):
        return False
    elif (
        has_sum
        and op == ">"
        and (
            (has_min and col_min >= 0 and col_sum <= val)
            or (has_max and col_max <= 0 and col_sum >= val)
        )
    ):
        return False
    elif (
        has_sum
        and op == ">="
        and (
            (has_min and col_min >= 0 and col_sum < val)
            or (has_max and col_max <= 0 and col_sum > val)
        )
    ):
        return False
    elif op == "in":
        if (has_max and col_max < min(val)) or (
            has_min and col_min > max(val)
        ):
            return False
        if all(_apply_filter_not_eq(elem, col_stats) for elem in val):
            return False
    elif op == "not in" and has_min and has_max:
        if any(elem == col_min == col_max for elem in val):
            return False
        col_range = None
        if isinstance(col_min, int):
            col_range = range(col_min, col_max)
        elif isinstance(col_min, datetime.datetime):
            col_range = pd.date_range(col_min, col_max)
        if col_range and all(elem in val for elem in col_range):
            return False
    return True


def _apply_filters(filters, stats):
    for conjunction in filters:
        if all(
            _apply_predicate(op, val, stats[col])
            for col, op, val in conjunction
        ):
            return True
    return False


@cuda.jit
def _apply_operator(minimums, maximums, op, other, range_value_pairs):
    # Load range and value
    range_idx, other_idx = cuda.grid(2)
    minimum = minimums[range_idx]
    maximum = maximums[range_idx]
    val = other[other_idx]
    # TODO: Download blocks of other into shared memory

    # Compute range-value pair
    if range_idx < minimums.size and other_idx < other.size:
        range_value_pairs[range_idx][other_idx] = not (
            (op == 0 and (minimum > val or maximum < val))
            or (op == 1 and (val == minimum == maximum))
            or (op == 2 and (minimum >= val))
            or (op == 3 and (minimum > val))
            or (op == 4 and (maximum < val))
            or (op == 5 and (maximum <= val))
        )


def _launch_filter_with_joins(other, minimums, maximums, op):
    # Initialize range-value pairs
    range_value_pairs = cupy.ndarray(
        [len(minimums), len(other)], dtype=np.bool
    )

    # Launch kernel to compute range-value pairs
    threadsperblock = 32
    blockspergrid = (
        (len(minimums) + (threadsperblock - 1)) // threadsperblock,
        (len(other) + (threadsperblock - 1)) // threadsperblock,
    )
    _apply_operator[blockspergrid, (threadsperblock, threadsperblock)](
        minimums, maximums, op, other, range_value_pairs
    )

    # Return result of boolean reduction for each range
    return cupy.any(range_value_pairs, 1)


def _filter_with_joins(minimums, maximums, op, other):
    assert len(minimums) == len(maximums)

    # Convert operator
    if op == "=" or op == "==":
        op = 0
    elif op == "!=":
        op = 1
    elif op == "<":
        op = 2
    elif op == "<=":
        op = 3
    elif op == ">=":
        op = 4
    elif op == ">":
        op = 5
    else:
        raise ValueError(
            '"{0}" is not a valid operator in join predicates.'.format(op)
        )

    # TODO: Handle dd.core.Series so that Dask dataframes don't have to be
    # .compute()ed (which involves downloading all partitions to client) and
    # can instead be used in place for filtering out stripes
    if isinstance(other, cudf.Series):
        return _launch_filter_with_joins(other, minimums, maximums, op)
    else:
        raise ValueError(
            "Joins must be with a cuDF or Dask cuDF series, not {0}.".format(
                type(op)
            )
        )


def _prepare_filters(filters):
    if filters is None:
        return None

    # Coerce filters into list of lists of tuples
    if isinstance(filters[0][0], str):
        filters = [filters]

    return filters
