# Copyright (c) 2018, NVIDIA CORPORATION.

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


def _apply_predicate(col, op, val, col_stats):
    # Sanitize operator
    if op not in {"=", "==", "!=", "<", "<=", ">", ">=", "in", "not in"}:
        raise ValueError(
            '"{0}" is not a valid operator in predicates.'.format(
                (col, op, val)
            )
        )

    has_min = "minimum" in col_stats
    has_max = "maximum" in col_stats
    has_sum = "sum" in col_stats
    col_min = col_stats["minimum"]
    col_max = col_stats["maximum"]
    col_sum = col_stats["maximum"]

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
    elif has_sum:
        if op == ">" and (
            (has_min and col_min >= 0 and col_sum <= val)
            or (has_max and col_max <= 0 and col_sum >= val)
        ):
            return False
        elif op == ">=" and (
            (has_min and col_min >= 0 and col_sum < val)
            or (has_max and col_max <= 0 and col_sum > val)
        ):
            return False
    elif op == "in":
        if (has_max and col_max < min(val)) or (
            has_min and col_min > max(val)
        ):
            return False
        if all([_apply_filter_not_eq(elem, col_stats) for elem in val]):
            return False
    elif (
        op == "not in"
        and has_min
        and has_max
        and any([elem >= col_min and elem <= col_max for elem in val])
        # TODO: Change this to only accept val is range or date range
    ):
        # return False if any elem in val == min == max
        # return False if all values between min and max is in val
        # create range of values from min to max
        # if val is a range, check that min-max is subset of val
        # if val is not range, check that every val between min and max is in val
        return False
    return True


def _apply_filters(filters, stats):
    for conjunction in filters:
        if all(
            [
                _apply_predicate(col, op, val, stats[col])
                for col, op, val in conjunction
            ]
        ):
            return True
    return False
