# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""
Utilities for hive-partitioned parquet scans with the GPU engine.

Polars raises ``NotImplementedError`` from ``view_current_node()`` when the GPU
engine callback encounters a hive-partitioned ``Scan`` node.  The API
intentionally signals that GPU engines must implement their own hive expansion.

``expand_hive_scan`` replaces the hive scan with an explicit list of parquet
files plus literal partition columns, producing a plan the GPU engine can
handle.  It also applies **partition pruning**: for simple equality and
comparison predicates on partition columns, only the matching partition
directories are included, avoiding unnecessary I/O.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Generator

import polars as pl

__all__: list[str] = ["expand_hive_scan"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_hive_scan(scan_node: dict) -> bool:
    """Return True when *scan_node* has ``hive_options.enabled = true``."""
    hive_opts = scan_node.get("unified_scan_args", {}).get("hive_options", {})
    return hive_opts.get("enabled") is True


def _get_hive_base_dir(scan_node: dict) -> Path:
    """Extract the base directory from a hive scan plan node."""
    paths = scan_node["sources"]["Paths"]
    return Path(paths[0]["inner"])


def _walk_hive_dir(
    base_dir: Path, prefix: dict[str, str] | None = None
) -> Generator[tuple[Path, dict[str, str]], None, None]:
    """
    Recursively walk a hive-partitioned directory structure.

    Yields ``(parquet_path, partition_vals)`` where *partition_vals* maps each
    partition column name to its raw string value (e.g. ``{"year": "2025"}``).
    Parquet files are yielded in sorted order for deterministic results.
    """
    if prefix is None:
        prefix = {}

    try:
        items = sorted(base_dir.iterdir())
    except (OSError, PermissionError):
        return

    parquet_files = [f for f in items if f.is_file() and f.suffix == ".parquet"]
    if parquet_files:
        # Leaf: this directory contains the actual data files.
        for pq_file in parquet_files:
            yield pq_file, prefix
        return

    for item in items:
        if item.is_dir() and "=" in item.name:
            col_name, col_val = item.name.split("=", 1)
            yield from _walk_hive_dir(item, {**prefix, col_name: col_val})


def _expr_from_plan_json(node: dict) -> pl.Expr:
    """
    Convert a polars plan JSON expression node to a ``pl.Expr``.

    Handles: ``Column``, ``Literal`` (Int/Float/Str/Bool/Date),
    ``BinaryExpr`` (comparison + logical operators), ``Alias``.

    Raises ``NotImplementedError`` for unsupported node types so that callers
    can fall back gracefully (treat predicate as always-True).
    """
    if "Column" in node:
        return pl.col(node["Column"])

    if "Literal" in node:
        lit = node["Literal"]
        if "Dyn" in lit:
            dyn = lit["Dyn"]
            if "Int" in dyn:
                return pl.lit(dyn["Int"])
            if "UInt" in dyn:
                return pl.lit(dyn["UInt"])
            if "Float" in dyn:
                return pl.lit(dyn["Float"])
            if "Str" in dyn:
                return pl.lit(dyn["Str"])
            if "Bool" in dyn:
                return pl.lit(dyn["Bool"])
            if "Null" in dyn:
                return pl.lit(None)
        # Polars uses a "Scalar" wrapper for string/bool/null literals
        if "Scalar" in lit:
            scalar = lit["Scalar"]
            if "String" in scalar:
                return pl.lit(scalar["String"])
            if "Boolean" in scalar:
                return pl.lit(scalar["Boolean"])
            if "Null" in scalar:
                return pl.lit(None)
            if "Int" in scalar:
                return pl.lit(scalar["Int"])
            if "UInt" in scalar:
                return pl.lit(scalar["UInt"])
            if "Float" in scalar:
                return pl.lit(scalar["Float"])
        if "Date" in lit:
            import datetime

            epoch = datetime.date(1970, 1, 1)
            return pl.lit(epoch + datetime.timedelta(days=lit["Date"])).cast(pl.Date)
        if "Datetime" in lit:
            # Polars serialises datetime as microseconds since epoch
            return pl.lit(lit["Datetime"][0], dtype=pl.Datetime("us"))

    if "BinaryExpr" in node:
        be = node["BinaryExpr"]
        left = _expr_from_plan_json(be["left"])
        right = _expr_from_plan_json(be["right"])
        op = be["op"]
        _OPS = {
            "Eq": lambda lhs, rhs: lhs == rhs,
            "NotEq": lambda lhs, rhs: lhs != rhs,
            "Lt": lambda lhs, rhs: lhs < rhs,
            "LtEq": lambda lhs, rhs: lhs <= rhs,
            "Gt": lambda lhs, rhs: lhs > rhs,
            "GtEq": lambda lhs, rhs: lhs >= rhs,
            "And": lambda lhs, rhs: lhs & rhs,
            "Or": lambda lhs, rhs: lhs | rhs,
        }
        if op in _OPS:
            return _OPS[op](left, right)
        raise NotImplementedError(f"Unsupported binary op in hive predicate: {op}")

    if "Alias" in node:
        alias_node = node["Alias"]
        if isinstance(alias_node, list):
            expr_node, name = alias_node[0], alias_node[1]
        else:
            expr_node, name = alias_node["expr"], alias_node["name"]
        return _expr_from_plan_json(expr_node).alias(name)

    raise NotImplementedError(
        f"Unsupported plan JSON predicate node type: {list(node.keys())}"
    )


def _partition_matches_predicate(
    partition_vals: dict[str, str],
    predicate_json: dict | None,
    schema: dict[str, pl.DataType],
) -> bool:
    """
    Evaluate a partition-column predicate to decide if a directory should be read.

    Only partition columns present in *partition_vals* are evaluated; predicates
    that reference non-partition columns are treated conservatively (True).
    Returns ``True`` to include the partition, ``False`` to prune it.
    """
    if predicate_json is None or not partition_vals:
        return True

    try:
        expr = _expr_from_plan_json(predicate_json)
    except NotImplementedError:
        return True  # Can't parse → be conservative

    # Build a one-row DataFrame containing the partition column values.
    part_data: dict[str, pl.Series] = {}
    for col, str_val in partition_vals.items():
        if col in schema:
            try:
                part_data[col] = pl.Series(col, [str_val]).cast(schema[col])
            except Exception:
                part_data[col] = pl.Series(col, [str_val])

    if not part_data:
        return True  # No partition cols in the one-row frame → can't evaluate

    try:
        df = pl.DataFrame(part_data)
        result = df.select(expr.alias("_check"))["_check"]
    except Exception:
        return True  # Any evaluation error → conservative
    else:
        if result.dtype == pl.Boolean:
            return bool(result[0])
        return True


def _find_hive_scan(
    plan_json: dict,
) -> tuple[dict | None, dict | None]:
    """
    Recursively locate the first hive-partitioned ``Scan`` in *plan_json*.

    Returns ``(scan_node, combined_predicate_json)`` where *combined_predicate_json*
    is the conjunction of all ``Filter`` predicates encountered on the path from
    the root down to the ``Scan``.  Returns ``(None, None)`` if no hive scan is
    found.
    """
    if "Scan" in plan_json:
        scan = plan_json["Scan"]
        if _is_hive_scan(scan):
            return scan, None
        return None, None

    if "Filter" in plan_json:
        filt = plan_json["Filter"]
        inner = filt["input"]
        if "Scan" in inner and _is_hive_scan(inner["Scan"]):
            return inner["Scan"], filt["predicate"]
        scan, existing_pred = _find_hive_scan(inner)
        if scan is not None:
            new_pred = filt["predicate"]
            if existing_pred is not None:
                combined = {
                    "BinaryExpr": {
                        "left": existing_pred,
                        "op": "And",
                        "right": new_pred,
                    }
                }
                return scan, combined
            return scan, new_pred

    # Recurse into other single-input plan nodes.
    _SINGLE_INPUT_KEYS = ("HStack", "Select", "SimpleProjection", "Slice", "Sort")
    for key in _SINGLE_INPUT_KEYS:
        if key in plan_json:
            node = plan_json[key]
            inner = node.get("input") or node.get("input_plan")
            if inner is not None:
                scan, pred = _find_hive_scan(inner)
                if scan is not None:
                    return scan, pred

    return None, None


def _build_expanded_scan(
    base_dir: Path,
    schema: dict[str, pl.DataType],
    predicate_json: dict | None = None,
) -> pl.LazyFrame:
    """
    Build a GPU-compatible ``LazyFrame`` from a hive-partitioned directory.

    For each partition directory, this function:

    1. Optionally prunes the partition based on *predicate_json* (avoids reading
       parquet files whose partition values cannot satisfy the predicate).
    2. Builds a ``scan_parquet`` for the explicit file path.
    3. Injects literal partition-column values using ``with_columns``.

    All partition columns are cast to the type declared in *schema* so that
    downstream filters and aggregations receive the correct dtypes.
    """
    partition_frames: list[pl.LazyFrame] = []

    for pq_file, partition_vals in _walk_hive_dir(base_dir):
        # Partition pruning: skip files whose partitions can't match the predicate.
        if not _partition_matches_predicate(partition_vals, predicate_json, schema):
            continue

        part_lf = pl.scan_parquet(str(pq_file))

        if partition_vals:
            lit_cols: list[pl.Expr] = []
            for col_name, str_val in partition_vals.items():
                if col_name in schema:
                    target_dtype = schema[col_name]
                    try:
                        typed_val = pl.Series("", [str_val]).cast(target_dtype)[0]
                        lit_cols.append(
                            pl.lit(typed_val, dtype=target_dtype).alias(col_name)
                        )
                    except Exception:
                        lit_cols.append(pl.lit(str_val).alias(col_name))
                else:
                    lit_cols.append(pl.lit(str_val).alias(col_name))
            part_lf = part_lf.with_columns(lit_cols)

        partition_frames.append(part_lf)

    if not partition_frames:
        # No matching partitions: return an empty frame with the expected schema.
        empty = {col: pl.Series(col, [], dtype=dtype) for col, dtype in schema.items()}
        return pl.from_dict(empty).lazy()

    return pl.concat(partition_frames, how="diagonal_relaxed")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def expand_hive_scan(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Transform a hive-partitioned parquet scan to a GPU-compatible ``LazyFrame``.

    Polars ``scan_parquet(..., hive_partitioning=True)`` uses directory-based
    partition discovery that the cudf-polars GPU engine cannot handle directly
    (Polars raises ``NotImplementedError`` from ``view_current_node()`` for
    hive-partitioned scans, signalling that GPU engines must expand them).

    ``expand_hive_scan`` replaces the hive scan with an equivalent plan that
    uses explicit file paths and injects partition column values as typed
    literals.  The GPU engine processes the result as a normal parquet scan.

    Partition pruning
    -----------------
    If a ``filter`` on a partition column is present in *lf*, the predicate is
    applied during directory discovery so that only matching partitions are
    opened.  Simple comparison predicates (``==``, ``!=``, ``<``, ``>``,
    ``<=``, ``>=``) and their ``&`` / ``|`` combinations are supported.
    Predicates that cannot be parsed conservatively include all partitions.

    Parameters
    ----------
    lf : pl.LazyFrame
        A LazyFrame produced by ``pl.scan_parquet(..., hive_partitioning=True)``,
        optionally followed by ``.filter()``, ``.select()``, or other operations.

    Returns
    -------
    pl.LazyFrame
        An equivalent ``LazyFrame`` using explicit ``scan_parquet`` calls with
        literal partition column injection.  The GPU engine can collect this
        without errors.

    Raises
    ------
    ValueError
        If no hive-partitioned scan is found in the plan.

    Examples
    --------
    Create a hive-partitioned directory structure and scan it with the GPU engine:

    >>> import polars as pl
    >>> from cudf_polars.io import expand_hive_scan
    >>>
    >>> lf = pl.scan_parquet("data/", hive_partitioning=True)
    >>> lf = lf.filter(pl.col("year") == 2025)
    >>> result = expand_hive_scan(lf).collect(engine=pl.GPUEngine(raise_on_fail=True))
    """
    buf = io.BytesIO()
    lf._ldf.serialize_json(buf)
    buf.seek(0)
    plan_json = json.loads(buf.read())

    scan_node, predicate_json = _find_hive_scan(plan_json)
    if scan_node is None:
        raise ValueError(
            "expand_hive_scan: no hive-partitioned scan found in the LazyFrame plan. "
            "Ensure the LazyFrame was created with "
            "scan_parquet(..., hive_partitioning=True)."
        )

    base_dir = _get_hive_base_dir(scan_node)
    schema = dict(lf.collect_schema())

    # Build the expanded plan with partition pruning applied during file discovery.
    expanded = _build_expanded_scan(base_dir, schema, predicate_json)

    # Re-apply the predicate on the expanded scan.  This handles:
    # (a) non-partition-column filters that couldn't be evaluated at directory
    #     discovery time, and
    # (b) within-partition row-level filtering (e.g. val > 5 on a data column).
    if predicate_json is not None:
        try:
            predicate_expr = _expr_from_plan_json(predicate_json)
            expanded = expanded.filter(predicate_expr)
        except NotImplementedError:
            # Complex predicate we can't parse — leave it to polars to apply.
            pass

    # Project to the output schema of the original LazyFrame.  This preserves
    # any column selection (e.g. .select("a", "year")) that was applied on top
    # of the hive scan before expand_hive_scan was called.
    return expanded.select(list(schema.keys()))


def _has_hive_scan(plan_json: dict) -> bool:
    """Return True if *plan_json* contains at least one hive-partitioned scan."""
    scan_node, _ = _find_hive_scan(plan_json)
    return scan_node is not None
