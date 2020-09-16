# Copyright (c) 2020, NVIDIA CORPORATION.


from numba import cuda
import numpy as np
import cupy
import datetime

import pandas as pd
import cudf
import dask.dataframe as dd


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
    elif isinstance(other, dd.core.Series):
        return _launch_filter_with_joins(
            other.compute(), minimums, maximums, op
        )


def _convert_datetime64_to_datetime(obj):
    if isinstance(obj, np.datetime64):
        return pd.to_datetime(obj).tz_localize("UTC")
    else:
        return obj


def _get_min(min_cache, val):
    if val in min_cache:
        return min_cache[id(val)]
    else:
        min_cache[id(val)] = _convert_datetime64_to_datetime(val.min())
        return min_cache[id(val)]


def _get_max(max_cache, val):
    if val in max_cache:
        return max_cache[id(val)]
    else:
        max_cache[id(val)] = _convert_datetime64_to_datetime(val.max())
        return max_cache[id(val)]


def _get_half_ranges(half_ranges_cache, val):
    # Check cache
    if id(val) in half_ranges_cache:
        return half_ranges_cache[id(val)]

    # Get position of larges change in value
    try:
        val_diff = val.diff()
        largest_jump_pos = val[val_diff == val_diff.max()].index[0]
    except (AssertionError, NotImplementedError):
        largest_jump_pos = len(val) // 2

    # Get ranges of each half of value
    first_half = val.loc[:largest_jump_pos].iloc[:-1]
    first_half_range = (
        _convert_datetime64_to_datetime(first_half.min()),
        _convert_datetime64_to_datetime(first_half.max()),
    )
    second_half = val.loc[largest_jump_pos:]
    second_half_range = (
        _convert_datetime64_to_datetime(second_half.min()),
        _convert_datetime64_to_datetime(second_half.max()),
    )

    half_ranges_cache[id(val)] = (first_half_range, second_half_range)
    return half_ranges_cache[id(val)]


def _load_to_local_arrow_array(loaded_values_cache, val):
    if id(val) in loaded_values_cache:
        return loaded_values_cache[id(val)]

    if isinstance(val, cudf.Series):
        loaded_values_cache[id(val)] = val.to_arrow()
        return val.to_arrow()
    elif isinstance(val, dd.core.Series):
        loaded_values_cache[id(val)] = val.compute().to_arrow()
        return val.compute().to_arrow()
    else:
        raise ValueError(
            "Expected cuDF or Dask cuDF series, not {0}.".format(type(val))
        )


def _apply_joins(filters, joins):
    # Caches
    half_ranges_cache = {}
    min_cache = {}
    max_cache = {}
    loaded_values_cache = {}

    # Modify filters for each join
    for col, op, val in joins:
        if op == "=" or op == "==":
            if len(val) < 128:
                val = _load_to_local_arrow_array(loaded_values_cache, val)
                for conjunction in filters:
                    conjunction.append((col, "in", val))
            else:
                fhr, shr = _get_half_ranges(half_ranges_cache, val)
                new_filters = []
                for conjunction in filters:
                    new_filters.append(
                        conjunction
                        + [(col, ">=", fhr[0]), (col, "<=", fhr[1])]
                    )
                    new_filters.append(
                        conjunction
                        + [(col, ">=", shr[0]), (col, "<=", shr[1])]
                    )
                filters = new_filters
        elif op == "!=":
            if len(val) < 128:
                val = _load_to_local_arrow_array(loaded_values_cache, val)
                for conjunction in filters:
                    conjunction.append((col, "not in", val))
            else:
                first_half_range, second_half_range = _get_half_ranges(val)
                if (
                    first_half_range[0] == first_half_range[1]
                    and second_half_range[0] == second_half_range[1]
                ):
                    if first_half_range[0] != second_half_range[0]:
                        for conjunction in filters:
                            conjunction.append(
                                (
                                    col,
                                    "not in",
                                    [
                                        first_half_range[0],
                                        second_half_range[0],
                                    ],
                                )
                            )
                    else:
                        for conjunction in filters:
                            conjunction.append(
                                (col, "!=", first_half_range[0])
                            )
        elif op == ">" or op == ">=":
            for conjunction in filters:
                conjunction.append((col, op, _get_min(min_cache, val)))
        elif op == "<" or op == "<=":
            for conjunction in filters:
                conjunction.append((col, op, _get_max(max_cache, val)))
        else:
            raise ValueError(
                '"{0}" is not a valid operator in join predicates.'.format(op)
            )

    return filters


def _prepare_filters(filters, joins):
    if filters is None:
        if joins is None:
            return None
        else:
            filters = [[]]
    else:
        # Coerce filters into list of lists of tuples
        if isinstance(filters[0][0], str):
            filters = [filters]

    # Add predicates to filters given joins
    if joins is not None:
        filters = _apply_joins(filters, joins)

    return filters
