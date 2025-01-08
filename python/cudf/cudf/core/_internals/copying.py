# Copyright (c) 2020-2024, NVIDIA CORPORATION.
from __future__ import annotations

from typing import TYPE_CHECKING

import pylibcudf as plc

import cudf
from cudf.core.buffer import acquire_spill_lock

if TYPE_CHECKING:
    from collections.abc import Iterable

    from cudf.core.column import ColumnBase
    from cudf.core.column.numerical import NumericalColumn


@acquire_spill_lock()
def gather(
    columns: Iterable[ColumnBase],
    gather_map: NumericalColumn,
    nullify: bool = False,
) -> list[ColumnBase]:
    plc_tbl = plc.copying.gather(
        plc.Table([col.to_pylibcudf(mode="read") for col in columns]),
        gather_map.to_pylibcudf(mode="read"),
        plc.copying.OutOfBoundsPolicy.NULLIFY
        if nullify
        else plc.copying.OutOfBoundsPolicy.DONT_CHECK,
    )
    return [
        cudf._lib.column.Column.from_pylibcudf(col)
        for col in plc_tbl.columns()
    ]


@acquire_spill_lock()
def scatter(
    sources: list[ColumnBase | cudf.Scalar],
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

    plc_tbl = plc.copying.scatter(
        plc.Table([col.to_pylibcudf(mode="read") for col in sources])  # type: ignore[union-attr]
        if isinstance(sources[0], cudf._lib.column.Column)
        else [slr.device_value.c_value for slr in sources],  # type: ignore[union-attr]
        scatter_map.to_pylibcudf(mode="read"),
        plc.Table([col.to_pylibcudf(mode="read") for col in target_columns]),
    )

    return [
        cudf._lib.column.Column.from_pylibcudf(col)
        for col in plc_tbl.columns()
    ]


@acquire_spill_lock()
def columns_split(
    input_columns: Iterable[ColumnBase], splits: list[int]
) -> list[list[ColumnBase]]:
    return [
        [
            cudf._lib.column.Column.from_pylibcudf(col)
            for col in plc_tbl.columns()
        ]
        for plc_tbl in plc.copying.split(
            plc.Table(
                [col.to_pylibcudf(mode="read") for col in input_columns]
            ),
            splits,
        )
    ]
