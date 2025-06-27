# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# TODO: remove need for this
# ruff: noqa: D101
"""DSL nodes for string operations."""

from __future__ import annotations

from enum import IntEnum, auto
from typing import TYPE_CHECKING, Any

from polars.exceptions import InvalidOperationError

import pylibcudf as plc

from cudf_polars.containers import Column
from cudf_polars.dsl.expressions.base import ExecutionContext, Expr
from cudf_polars.dsl.expressions.literal import Literal, LiteralColumn
from cudf_polars.dsl.utils.reshape import broadcast

if TYPE_CHECKING:
    from typing_extensions import Self

    from polars.polars import _expr_nodes as pl_expr

    from cudf_polars.containers import DataFrame, DataType

__all__ = ["StringFunction"]


class StringFunction(Expr):
    class Name(IntEnum):
        """Internal and picklable representation of polars' `StringFunction`."""

        Base64Decode = auto()
        Base64Encode = auto()
        ConcatHorizontal = auto()
        ConcatVertical = auto()
        Contains = auto()
        ContainsAny = auto()
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
        Normalize = auto()
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
        dtype: DataType,
        name: StringFunction.Name,
        options: tuple[Any, ...],
        *children: Expr,
    ) -> None:
        self.dtype = dtype
        self.options = options
        self.name = name
        self.children = children
        self.is_pointwise = self.name != StringFunction.Name.ConcatVertical
        self._validate_input()

    def _validate_input(self) -> None:
        if self.name not in (
            StringFunction.Name.ConcatHorizontal,
            StringFunction.Name.ConcatVertical,
            StringFunction.Name.Contains,
            StringFunction.Name.EndsWith,
            StringFunction.Name.Head,
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
            StringFunction.Name.Reverse,
            StringFunction.Name.Tail,
            StringFunction.Name.Titlecase,
        ):
            raise NotImplementedError(f"String function {self.name!r}")
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
                pattern = self.children[1].value
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
            # Above, we raise NotImplementedError if the target is not a Literal,
            # so we can safely access .value here.
            if target.value == "":  # type: ignore[attr-defined]
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
            # Above, we raise NotImplementedError if the target is not a Literal,
            # so we can safely access .value here.
            if (isinstance(target, Literal) and target.value == "") or (
                isinstance(target, LiteralColumn) and (target.value == "").any()
            ):
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
        self, df: DataFrame, *, context: ExecutionContext = ExecutionContext.FRAME
    ) -> Column:
        """Evaluate this expression given a dataframe for context."""
        if self.name is StringFunction.Name.ConcatHorizontal:
            columns = [
                Column(
                    child.evaluate(df, context=context).obj, dtype=child.dtype
                ).astype(self.dtype)
                for child in self.children
            ]

            broadcasted = broadcast(
                *columns, target_length=max(col.size for col in columns)
            )

            delimiter, ignore_nulls = self.options

            return Column(
                plc.strings.combine.concatenate(
                    plc.Table([col.obj for col in broadcasted]),
                    plc.Scalar.from_py(delimiter, self.dtype.plc),
                    None if ignore_nulls else plc.Scalar.from_py(None, self.dtype.plc),
                    None,
                    plc.strings.combine.SeparatorOnNulls.NO,
                ),
                dtype=self.dtype,
            )
        elif self.name is StringFunction.Name.ConcatVertical:
            (child,) = self.children
            column = child.evaluate(df, context=context)
            delimiter, ignore_nulls = self.options
            if column.null_count > 0 and not ignore_nulls:
                return Column(plc.Column.all_null_like(column.obj, 1), dtype=self.dtype)
            return Column(
                plc.strings.combine.join_strings(
                    column.obj,
                    plc.Scalar.from_py(delimiter, self.dtype.plc),
                    plc.Scalar.from_py(None, self.dtype.plc),
                ),
                dtype=self.dtype,
            )
        elif self.name is StringFunction.Name.Contains:
            child, arg = self.children
            column = child.evaluate(df, context=context)

            literal, _ = self.options
            if literal:
                pat = arg.evaluate(df, context=context)
                pattern = (
                    pat.obj_scalar
                    if pat.is_scalar and pat.size != column.size
                    else pat.obj
                )
                return Column(
                    plc.strings.find.contains(column.obj, pattern), dtype=self.dtype
                )
            else:
                return Column(
                    plc.strings.contains.contains_re(column.obj, self._regex_program),
                    dtype=self.dtype,
                )
        elif self.name is StringFunction.Name.Slice:
            child, expr_offset, expr_length = self.children
            assert isinstance(expr_offset, Literal)
            assert isinstance(expr_length, Literal)

            column = child.evaluate(df, context=context)
            # libcudf slices via [start,stop).
            # polars slices with offset + length where start == offset
            # stop = start + length. Negative values for start look backward
            # from the last element of the string. If the end index would be
            # below zero, an empty string is returned.
            # Do this maths on the host
            start = expr_offset.value
            length = expr_length.value

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
                    plc.Scalar.from_py(start, plc.DataType(plc.TypeId.INT32)),
                    plc.Scalar.from_py(stop, plc.DataType(plc.TypeId.INT32)),
                ),
                dtype=self.dtype,
            )
        elif self.name in {
            StringFunction.Name.StripChars,
            StringFunction.Name.StripCharsStart,
            StringFunction.Name.StripCharsEnd,
        }:
            column, chars = (c.evaluate(df, context=context) for c in self.children)
            if self.name is StringFunction.Name.StripCharsStart:
                side = plc.strings.SideType.LEFT
            elif self.name is StringFunction.Name.StripCharsEnd:
                side = plc.strings.SideType.RIGHT
            else:
                side = plc.strings.SideType.BOTH
            return Column(
                plc.strings.strip.strip(column.obj, side, chars.obj_scalar),
                dtype=self.dtype,
            )

        elif self.name is StringFunction.Name.Tail:
            column = self.children[0].evaluate(df, context=context)

            assert isinstance(self.children[1], Literal)
            if self.children[1].value is None:
                return Column(
                    plc.Column.from_scalar(
                        plc.Scalar.from_py(None, self.dtype.plc),
                        column.size,
                    ),
                    self.dtype,
                )
            elif self.children[1].value == 0:
                result = plc.Column.from_scalar(
                    plc.Scalar.from_py("", self.dtype.plc),
                    column.size,
                )
                if column.obj.null_mask():
                    result = result.with_mask(
                        column.obj.null_mask(), column.obj.null_count()
                    )
                return Column(result, self.dtype)

            else:
                start = -(self.children[1].value)
                end = 2**31 - 1
                return Column(
                    plc.strings.slice.slice_strings(
                        column.obj,
                        plc.Scalar.from_py(start, plc.DataType(plc.TypeId.INT32)),
                        plc.Scalar.from_py(end, plc.DataType(plc.TypeId.INT32)),
                        None,
                    ),
                    self.dtype,
                )
        elif self.name is StringFunction.Name.Head:
            column = self.children[0].evaluate(df, context=context)

            assert isinstance(self.children[1], Literal)

            end = self.children[1].value
            if end is None:
                return Column(
                    plc.Column.from_scalar(
                        plc.Scalar.from_py(None, self.dtype.plc),
                        column.size,
                    ),
                    self.dtype,
                )
            return Column(
                plc.strings.slice.slice_strings(
                    column.obj,
                    plc.Scalar.from_py(0, plc.DataType(plc.TypeId.INT32)),
                    plc.Scalar.from_py(end, plc.DataType(plc.TypeId.INT32)),
                ),
                self.dtype,
            )

        columns = [child.evaluate(df, context=context) for child in self.children]
        if self.name is StringFunction.Name.Lowercase:
            (column,) = columns
            return Column(plc.strings.case.to_lower(column.obj), dtype=self.dtype)
        elif self.name is StringFunction.Name.Uppercase:
            (column,) = columns
            return Column(plc.strings.case.to_upper(column.obj), dtype=self.dtype)
        elif self.name is StringFunction.Name.EndsWith:
            column, suffix = columns
            return Column(
                plc.strings.find.ends_with(
                    column.obj,
                    suffix.obj_scalar
                    if column.size != suffix.size and suffix.is_scalar
                    else suffix.obj,
                ),
                dtype=self.dtype,
            )
        elif self.name is StringFunction.Name.StartsWith:
            column, prefix = columns
            return Column(
                plc.strings.find.starts_with(
                    column.obj,
                    prefix.obj_scalar
                    if column.size != prefix.size and prefix.is_scalar
                    else prefix.obj,
                ),
                dtype=self.dtype,
            )
        elif self.name is StringFunction.Name.Strptime:
            # TODO: ignores ambiguous
            format, strict, exact, cache = self.options
            col = self.children[0].evaluate(df, context=context)

            is_timestamps = plc.strings.convert.convert_datetime.is_timestamp(
                col.obj, format
            )

            if strict:
                if not plc.reduce.reduce(
                    is_timestamps,
                    plc.aggregation.all(),
                    plc.DataType(plc.TypeId.BOOL8),
                ).to_py():
                    raise InvalidOperationError("conversion from `str` failed.")
            else:
                not_timestamps = plc.unary.unary_operation(
                    is_timestamps, plc.unary.UnaryOperator.NOT
                )
                null = plc.Scalar.from_py(None, col.obj.type())
                res = plc.copying.boolean_mask_scatter(
                    [null], plc.Table([col.obj]), not_timestamps
                )
                return Column(
                    plc.strings.convert.convert_datetime.to_timestamps(
                        res.columns()[0], self.dtype.plc, format
                    ),
                    dtype=self.dtype,
                )
        elif self.name is StringFunction.Name.Replace:
            column, target, repl = columns
            n, _ = self.options
            return Column(
                plc.strings.replace.replace(
                    column.obj, target.obj_scalar, repl.obj_scalar, maxrepl=n
                ),
                dtype=self.dtype,
            )
        elif self.name is StringFunction.Name.ReplaceMany:
            column, target, repl = columns
            return Column(
                plc.strings.replace.replace_multiple(column.obj, target.obj, repl.obj),
                dtype=self.dtype,
            )
        elif self.name is StringFunction.Name.Reverse:
            (column,) = columns
            return Column(plc.strings.reverse.reverse(column.obj), dtype=self.dtype)
        elif self.name is StringFunction.Name.Titlecase:
            (column,) = columns
            return Column(plc.strings.capitalize.title(column.obj), dtype=self.dtype)
        raise NotImplementedError(
            f"StringFunction {self.name}"
        )  # pragma: no cover; handled by init raising
