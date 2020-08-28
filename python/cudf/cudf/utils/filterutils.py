# Copyright (c) 2018, NVIDIA CORPORATION.

import pandas as pd


def _apply_filter_bool_eq(val, col_stats):
    if col_stats["kind"] == 0:
        if val is True:
            if (
                "true_count" in col_stats and col_stats["true_count"] == 0
            ) or (
                "false_count" in col_stats
                and col_stats["false_count"] == col_stats["number_of_values"]
            ):
                return False
        elif val is False:
            if (
                "false_count" in col_stats and col_stats["false_count"] == 0
            ) or (
                "true_count" in col_stats
                and col_stats["true_count"] == col_stats["number_of_values"]
            ):
                return False
    return True


def _apply_filter_not_eq(val, col_stats):
    return (
        ("minimum" in col_stats and val < col_stats["minimum"])
        or ("lower_bound" in col_stats and val < col_stats["lower_bound"])
        or ("maximum" in col_stats and val > col_stats["maximum"])
        or ("upper_bound" in col_stats and val > col_stats["upper_bound"])
    )


def _apply_predicate(col, op, val, col_stats):
    # Sanitize operator
    if op not in {"=", "==", "!=", "<", "<=", ">", ">=", "in", "not in"}:
        raise ValueError(
            '"{0}" is not a valid operator in predicates.'.format(
                (col, op, val)
            )
        )

    # Apply operator
    if op == "=" or op == "==":
        if _apply_filter_not_eq(val, col_stats):
            return False
        if (
            "has_null" in col_stats
            and pd.isnull(val)
            and col_stats["has_null"]
        ):
            return False
        if not _apply_filter_bool_eq(val, col_stats):
            return False
    elif op == "!=":
        if (
            "minimum" in col_stats
            and "maximum" in col_stats
            and val == col_stats["minimum"]
            and val == col_stats["maximum"]
        ):
            return False
        if _apply_filter_bool_eq(val, col_stats):
            return False
    elif (
        op == "<" and "minimum" in col_stats and val <= col_stats["minimum"]
    ) or (
        op == "<=" and "minimum" in col_stats and val < col_stats["minimum"]
    ):
        return False
    elif op == ">":
        if "maximum" in col_stats and val >= col_stats["maximum"]:
            return False
        if (
            "sum" in col_stats
            and "minimum" in col_stats
            and col_stats["minimum"] >= 0
            and col_stats["sum"] <= val
        ) or (
            "sum" in col_stats
            and "minimum" in col_stats
            and col_stats["maximum"] <= 0
            and col_stats["sum"] >= val
        ):
            return False
    elif op == ">=":
        if "maximum" in col_stats and val > col_stats["maximum"]:
            return False
        if (
            "sum" in col_stats
            and "minimum" in col_stats
            and col_stats["minimum"] >= 0
            and col_stats["sum"] < val
        ) or (
            "sum" in col_stats
            and "maximum" in col_stats
            and col_stats["maximum"] <= 0
            and col_stats["sum"] > val
        ):
            return False
    elif op == "in":
        if ("maximum" in col_stats and col_stats["maximum"] < min(val)) or (
            "minimum" in col_stats and col_stats["minimum"] > max(val)
        ):
            return False
        if all([_apply_filter_not_eq(elem, col_stats) for elem in val]):
            return False
    elif (
        op == "not in"
        and "minimum" in col_stats
        and "maximum" in col_stats
        and any(
            [
                elem >= col_stats["minimum"] and elem <= col_stats["maximum"]
                for elem in val
            ]
        )
    ):
        return False
    return True


def _apply_filters(filters, stats):
    for conjunction in filters:
        res = True
        for col, op, val in conjunction:
            # Get stats
            col_stats = stats[col]
            if "lower_bound" in col_stats:
                col_stats["minimum"] = col_stats["lower_bound"]
            if "upper_bound" in col_stats:
                col_stats["maximum"] = col_stats["upper_bound"]

            # Apply operators
            if not _apply_predicate(col, op, val, col_stats):
                res = False
        if res:
            return True
    return False
