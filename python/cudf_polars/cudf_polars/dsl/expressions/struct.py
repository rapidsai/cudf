# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# TODO: Document StructFunction to remove noqa
# ruff: noqa: D101
"""Struct DSL nodes."""

from __future__ import annotations

from enum import IntEnum, auto
from io import StringIO
from typing import TYPE_CHECKING, Any, ClassVar

import pylibcudf as plc

from cudf_polars.containers import Column
from cudf_polars.dsl.expressions.base import ExecutionContext, Expr

if TYPE_CHECKING:
    from typing_extensions import Self

    from polars.polars import _expr_nodes as pl_expr

    from cudf_polars.containers import DataFrame, DataType

__all__ = ["StructFunction"]


class StructFunction(Expr):
    class Name(IntEnum):
        """Internal and picklable representation of polars' `StructFunction`."""

        FieldByIndex = auto()
        FieldByName = auto()
        RenameFields = auto()
        PrefixFields = auto()
        SuffixFields = auto()
        JsonEncode = auto()
        WithFields = auto()  # TODO: https://github.com/rapidsai/cudf/issues/19284
        MapFieldNames = auto()  # TODO: https://github.com/rapidsai/cudf/issues/19285
        MultipleFields = (
            auto()
        )  # https://github.com/pola-rs/polars/pull/23022#issuecomment-2933910958

        @classmethod
        def from_polars(cls, obj: pl_expr.StructFunction) -> Self:
            """Convert from polars' `StructFunction`."""
            try:
                function, name = str(obj).split(".", maxsplit=1)
            except ValueError:
                # Failed to unpack string
                function = None
            if function != "StructFunction":
                raise ValueError("StructFunction required")
            return getattr(cls, name)

    __slots__ = ("name", "options")
    _non_child = ("dtype", "name", "options")

    _valid_ops: ClassVar[set[Name]] = {
        Name.FieldByIndex,
        Name.FieldByName,
        Name.RenameFields,
        Name.PrefixFields,
        Name.SuffixFields,
        Name.JsonEncode,
    }

    def __init__(
        self,
        dtype: DataType,
        name: StructFunction.Name,
        options: tuple[Any, ...],
        *children: Expr,
    ) -> None:
        self.dtype = dtype
        self.options = options
        self.name = name
        self.children = children
        self.is_pointwise = True
        if self.name not in self._valid_ops:
            raise NotImplementedError(
                f"Struct function {self.name}"
            )  # pragma: no cover

    def do_evaluate(
        self, df: DataFrame, *, context: ExecutionContext = ExecutionContext.FRAME
    ) -> Column:
        """Evaluate this expression given a dataframe for context."""
        columns = [child.evaluate(df, context=context) for child in self.children]
        (column,) = columns
        if self.name == StructFunction.Name.FieldByName:
            field_index = next(
                (
                    i
                    for i, field in enumerate(self.children[0].dtype.polars.fields)
                    if field.name == self.options[0]
                ),
                None,
            )
            assert field_index is not None
            return Column(
                column.obj.children()[field_index],
                dtype=self.dtype,
            )
        elif self.name == StructFunction.Name.JsonEncode:
            # Once https://github.com/rapidsai/cudf/issues/19338 is implemented,
            # we can use do this conversion on host.
            buff = StringIO()
            target = plc.io.SinkInfo([buff])
            table = plc.Table(column.obj.children())
            metadata = plc.io.TableWithMetadata(
                table,
                [(field.name, []) for field in self.children[0].dtype.polars.fields],
            )
            options = (
                plc.io.json.JsonWriterOptions.builder(target, table)
                .lines(val=True)
                .na_rep("null")
                .include_nulls(val=True)
                .metadata(metadata)
                .utf8_escaped(val=False)
                .build()
            )
            plc.io.json.write_json(options)
            return Column(
                plc.Column.from_iterable_of_py(buff.getvalue().split()),
                dtype=self.dtype,
            )
        elif self.name in {
            StructFunction.Name.RenameFields,
            StructFunction.Name.PrefixFields,
            StructFunction.Name.SuffixFields,
        }:
            return column
        else:
            raise NotImplementedError(
                f"Struct function {self.name}"
            )  # pragma: no cover
