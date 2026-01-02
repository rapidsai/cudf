# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from contextlib import ExitStack
from typing import TYPE_CHECKING, cast

import pylibcudf as plc

if TYPE_CHECKING:
    from collections.abc import Iterable

    from cudf.core.column import ColumnBase
    from cudf.core.column.numerical import NumericalColumn


def gather(
    columns: Iterable[ColumnBase],
    gather_map: NumericalColumn,
    nullify: bool = False,
) -> list[plc.Column]:
    with ExitStack() as stack:
        for col in columns:
            stack.enter_context(col.access(mode="read", scope="internal"))
        stack.enter_context(gather_map.access(mode="read", scope="internal"))

        plc_tbl = plc.copying.gather(
            plc.Table([col.plc_column for col in columns]),
            gather_map.plc_column,
            plc.copying.OutOfBoundsPolicy.NULLIFY
            if nullify
            else plc.copying.OutOfBoundsPolicy.DONT_CHECK,
        )
        return plc_tbl.columns()


def scatter(
    sources: list[ColumnBase] | list[plc.Scalar],
    scatter_map: NumericalColumn,
    target_columns: list[ColumnBase],
    bounds_check: bool = True,
):
    """
    Scattering source into target as per the scatter map.
    `source` can be a list of scalars, or a list of columns. The number of
    items in `sources` must equal the number of `target_columns` to scatter.
    """
    # TODO: Only single column scatter is used, we should explore multi-column
    # scatter for frames for performance increase.

    if len(sources) != len(target_columns):
        raise ValueError("Mismatched number of source and target columns.")

    if len(sources) == 0:
        return []

    if bounds_check:
        n_rows = len(target_columns[0])
        if not (
            (scatter_map >= -n_rows).all() and (scatter_map < n_rows).all()
        ):
            raise IndexError(
                f"index out of bounds for column of size {n_rows}"
            )

    stack = ExitStack()
    with stack:
        for col in target_columns:
            stack.enter_context(col.access(mode="write"))
        plc_tbl = plc.copying.scatter(
            cast(list[plc.Scalar], sources)
            if isinstance(sources[0], plc.Scalar)
            else plc.Table(
                [
                    col.plc_column  # type: ignore[union-attr]
                    for col in sources
                ]
            ),
            scatter_map.plc_column,
            plc.Table([col.plc_column for col in target_columns]),
        )

    return plc_tbl.columns()


def columns_split(
    input_columns: Iterable[ColumnBase], splits: list[int]
) -> list[list[plc.Column]]:
    with ExitStack() as stack:
        cols_list = list(input_columns)
        for col in cols_list:
            stack.enter_context(col.access(mode="read", scope="internal"))

        return [
            plc_tbl.columns()
            for plc_tbl in plc.copying.split(
                plc.Table([col.plc_column for col in cols_list]),
                splits,
            )
        ]
