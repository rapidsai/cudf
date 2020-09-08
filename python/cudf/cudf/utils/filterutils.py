# Copyright (c) 2018, NVIDIA CORPORATION.

import datetime

import pandas as pd


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
        if pd.isnull(val) and col_stats["has_null"]:
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


def _columns_in_predicate(filters):
    return [col for conjunction in filters for (col, op, val) in conjunction]


def _predicate_to_query(filters):
    query = ""
    vals = {}

    first_conjunction = True
    for conjunction in filters:
        if first_conjunction:
            query += "("
            first_conjunction = False
        else:
            query += " or ("
        first_predicate = True
        for col, op, val in conjunction:
            if first_predicate:
                query += "("
                first_predicate = False
            else:
                query += " and ("
            query += col + " " + op + " @" + str(len(vals))
            vals[str(len(vals))] = val
            query += ")"
        query += ")"

    return query, vals


def _prepare_filters(filters):
    # Coerce filters into list of lists of tuples
    if isinstance(filters[0][0], str):
        filters = [filters]

    return filters
