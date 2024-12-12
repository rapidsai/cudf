# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# TODO: remove need for this
# ruff: noqa: D101
"""DSL nodes for string operations."""

from __future__ import annotations

from enum import IntEnum, auto
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow.compute as pc

from polars.exceptions import InvalidOperationError

import pylibcudf as plc

from cudf_polars.containers import Column
from cudf_polars.dsl.expressions.base import ExecutionContext, Expr
from cudf_polars.dsl.expressions.literal import Literal, LiteralColumn

if TYPE_CHECKING:
    from collections.abc import Mapping

    from typing_extensions import Self

    from polars.polars import _expr_nodes as pl_expr

    from cudf_polars.containers import DataFrame

__all__ = ["StringFunction"]


class StringFunction(Expr):
    class Name(IntEnum):
        """Internal and picklable representation of polars' `StringFunction`."""

        Base64Decode = auto()
        Base64Encode = auto()
        ConcatHorizontal = auto()
        ConcatVertical = auto()
        Contains = auto()
        ContainsMany = auto()
        CountMatches = auto()
        EndsWith = auto()
        EscapeRegex = auto()
        Extract = auto()
        ExtractAll = auto()
        ExtractGroups = auto()
        Find = auto()
        Head = auto()
        HexDecode = auto()
        HexEncode = auto()
        JsonDecode = auto()
        JsonPathMatch = auto()
        LenBytes = auto()
        LenChars = auto()
        Lowercase = auto()
        PadEnd = auto()
        PadStart = auto()
        Replace = auto()
        ReplaceMany = auto()
        Reverse = auto()
        Slice = auto()
        Split = auto()
        SplitExact = auto()
        SplitN = auto()
        StartsWith = auto()
        StripChars = auto()
        StripCharsEnd = auto()
        StripCharsStart = auto()
        StripPrefix = auto()
        StripSuffix = auto()
        Strptime = auto()
        Tail = auto()
        Titlecase = auto()
        ToDecimal = auto()
        ToInteger = auto()
        Uppercase = auto()
        ZFill = auto()

        @classmethod
        def from_polars(cls, obj: pl_expr.StringFunction) -> Self:
            """Convert from polars' `StringFunction`."""
            try:
                function, name = str(obj).split(".", maxsplit=1)
            except ValueError:
                # Failed to unpack string
                function = None
            if function != "StringFunction":
                raise ValueError("StringFunction required")
            return getattr(cls, name)

    __slots__ = ("_regex_program", "name", "options")
    _non_child = ("dtype", "name", "options")

    def __init__(
        self,
        dtype: plc.DataType,
        name: StringFunction.Name,
        options: tuple[Any, ...],
        *children: Expr,
    ) -> None:
        self.dtype = dtype
        self.options = options
        self.name = name
        self.children = children
        self.is_pointwise = True
        self._validate_input()

    def _validate_input(self):
        if self.name not in (
            StringFunction.Name.Contains,
            StringFunction.Name.EndsWith,
            StringFunction.Name.Lowercase,
            StringFunction.Name.Replace,
            StringFunction.Name.ReplaceMany,
            StringFunction.Name.Slice,
            StringFunction.Name.Strptime,
            StringFunction.Name.StartsWith,
            StringFunction.Name.StripChars,
            StringFunction.Name.StripCharsStart,
            StringFunction.Name.StripCharsEnd,
            StringFunction.Name.Uppercase,
        ):
            raise NotImplementedError(f"String function {self.name}")
        if self.name is StringFunction.Name.Contains:
            literal, strict = self.options
            if not literal:
                if not strict:
                    raise NotImplementedError(
                        "f{strict=} is not supported for regex contains"
                    )
                if not isinstance(self.children[1], Literal):
                    raise NotImplementedError(
                        "Regex contains only supports a scalar pattern"
                    )
                pattern = self.children[1].value.as_py()
                try:
                    self._regex_program = plc.strings.regex_program.RegexProgram.create(
                        pattern,
                        flags=plc.strings.regex_flags.RegexFlags.DEFAULT,
                    )
                except RuntimeError as e:
                    raise NotImplementedError(
                        f"Unsupported regex {pattern} for GPU engine."
                    ) from e
        elif self.name is StringFunction.Name.Replace:
            _, literal = self.options
            if not literal:
                raise NotImplementedError("literal=False is not supported for replace")
            if not all(isinstance(expr, Literal) for expr in self.children[1:]):
                raise NotImplementedError("replace only supports scalar target")
            target = self.children[1]
            if target.value == pa.scalar("", type=pa.string()):
                raise NotImplementedError(
                    "libcudf replace does not support empty strings"
                )
        elif self.name is StringFunction.Name.ReplaceMany:
            (ascii_case_insensitive,) = self.options
            if ascii_case_insensitive:
                raise NotImplementedError(
                    "ascii_case_insensitive not implemented for replace_many"
                )
            if not all(
                isinstance(expr, (LiteralColumn, Literal)) for expr in self.children[1:]
            ):
                raise NotImplementedError("replace_many only supports literal inputs")
            target = self.children[1]
            if pc.any(pc.equal(target.value, "")).as_py():
                raise NotImplementedError(
                    "libcudf replace_many is implemented differently from polars "
                    "for empty strings"
                )
        elif self.name is StringFunction.Name.Slice:
            if not all(isinstance(child, Literal) for child in self.children[1:]):
                raise NotImplementedError(
                    "Slice only supports literal start and stop values"
                )
        elif self.name is StringFunction.Name.Strptime:
            format, _, exact, cache = self.options
            if cache:
                raise NotImplementedError("Strptime cache is a CPU feature")
            if format is None:
                raise NotImplementedError("Strptime format is required")
            if not exact:
                raise NotImplementedError("Strptime does not support exact=False")
        elif self.name in {
            StringFunction.Name.StripChars,
            StringFunction.Name.StripCharsStart,
            StringFunction.Name.StripCharsEnd,
        }:
            if not isinstance(self.children[1], Literal):
                raise NotImplementedError(
                    "strip operations only support scalar patterns"
                )

    def do_evaluate(
        self,
        df: DataFrame,
        *,
        context: ExecutionContext = ExecutionContext.FRAME,
        mapping: Mapping[Expr, Column] | None = None,
    ) -> Column:
        """Evaluate this expression given a dataframe for context."""
        if self.name is StringFunction.Name.Contains:
            child, arg = self.children
            column = child.evaluate(df, context=context, mapping=mapping)

            literal, _ = self.options
            if literal:
                pat = arg.evaluate(df, context=context, mapping=mapping)
                pattern = (
                    pat.obj_scalar
                    if pat.is_scalar and pat.obj.size() != column.obj.size()
                    else pat.obj
                )
                return Column(plc.strings.find.contains(column.obj, pattern))
            else:
                return Column(
                    plc.strings.contains.contains_re(column.obj, self._regex_program)
                )
        elif self.name is StringFunction.Name.Slice:
            child, expr_offset, expr_length = self.children
            assert isinstance(expr_offset, Literal)
            assert isinstance(expr_length, Literal)

            column = child.evaluate(df, context=context, mapping=mapping)
            # libcudf slices via [start,stop).
            # polars slices with offset + length where start == offset
            # stop = start + length. Negative values for start look backward
            # from the last element of the string. If the end index would be
            # below zero, an empty string is returned.
            # Do this maths on the host
            start = expr_offset.value.as_py()
            length = expr_length.value.as_py()

            if length == 0:
                stop = start
            else:
                # No length indicates a scan to the end
                # The libcudf equivalent is a null stop
                stop = start + length if length else None
                if length and start < 0 and length >= -start:
                    stop = None
            return Column(
                plc.strings.slice.slice_strings(
                    column.obj,
                    plc.interop.from_arrow(pa.scalar(start, type=pa.int32())),
                    plc.interop.from_arrow(pa.scalar(stop, type=pa.int32())),
                )
            )
        elif self.name in {
            StringFunction.Name.StripChars,
            StringFunction.Name.StripCharsStart,
            StringFunction.Name.StripCharsEnd,
        }:
            column, chars = (
                c.evaluate(df, context=context, mapping=mapping) for c in self.children
            )
            if self.name is StringFunction.Name.StripCharsStart:
                side = plc.strings.SideType.LEFT
            elif self.name is StringFunction.Name.StripCharsEnd:
                side = plc.strings.SideType.RIGHT
            else:
                side = plc.strings.SideType.BOTH
            return Column(plc.strings.strip.strip(column.obj, side, chars.obj_scalar))

        columns = [
            child.evaluate(df, context=context, mapping=mapping)
            for child in self.children
        ]
        if self.name is StringFunction.Name.Lowercase:
            (column,) = columns
            return Column(plc.strings.case.to_lower(column.obj))
        elif self.name is StringFunction.Name.Uppercase:
            (column,) = columns
            return Column(plc.strings.case.to_upper(column.obj))
        elif self.name is StringFunction.Name.EndsWith:
            column, suffix = columns
            return Column(
                plc.strings.find.ends_with(
                    column.obj,
                    suffix.obj_scalar
                    if column.obj.size() != suffix.obj.size() and suffix.is_scalar
                    else suffix.obj,
                )
            )
        elif self.name is StringFunction.Name.StartsWith:
            column, prefix = columns
            return Column(
                plc.strings.find.starts_with(
                    column.obj,
                    prefix.obj_scalar
                    if column.obj.size() != prefix.obj.size() and prefix.is_scalar
                    else prefix.obj,
                )
            )
        elif self.name is StringFunction.Name.Strptime:
            # TODO: ignores ambiguous
            format, strict, exact, cache = self.options
            col = self.children[0].evaluate(df, context=context, mapping=mapping)

            is_timestamps = plc.strings.convert.convert_datetime.is_timestamp(
                col.obj, format
            )

            if strict:
                if not plc.interop.to_arrow(
                    plc.reduce.reduce(
                        is_timestamps,
                        plc.aggregation.all(),
                        plc.DataType(plc.TypeId.BOOL8),
                    )
                ).as_py():
                    raise InvalidOperationError("conversion from `str` failed.")
            else:
                not_timestamps = plc.unary.unary_operation(
                    is_timestamps, plc.unary.UnaryOperator.NOT
                )

                null = plc.interop.from_arrow(pa.scalar(None, type=pa.string()))
                res = plc.copying.boolean_mask_scatter(
                    [null], plc.Table([col.obj]), not_timestamps
                )
                return Column(
                    plc.strings.convert.convert_datetime.to_timestamps(
                        res.columns()[0], self.dtype, format
                    )
                )
        elif self.name is StringFunction.Name.Replace:
            column, target, repl = columns
            n, _ = self.options
            return Column(
                plc.strings.replace.replace(
                    column.obj, target.obj_scalar, repl.obj_scalar, maxrepl=n
                )
            )
        elif self.name is StringFunction.Name.ReplaceMany:
            column, target, repl = columns
            return Column(
                plc.strings.replace.replace_multiple(column.obj, target.obj, repl.obj)
            )
        raise NotImplementedError(
            f"StringFunction {self.name}"
        )  # pragma: no cover; handled by init raising
