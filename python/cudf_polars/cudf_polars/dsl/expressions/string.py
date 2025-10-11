# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# TODO: Document StringFunction to remove noqa
# ruff: noqa: D101
"""DSL nodes for string operations."""

from __future__ import annotations

import functools
import re
from datetime import datetime
from enum import IntEnum, auto
from typing import TYPE_CHECKING, Any, ClassVar

from polars.exceptions import InvalidOperationError
from polars.polars import dtype_str_repr

import pylibcudf as plc

from cudf_polars.containers import Column
from cudf_polars.dsl.expressions.base import ExecutionContext, Expr
from cudf_polars.dsl.expressions.literal import Literal, LiteralColumn
from cudf_polars.dsl.utils.reshape import broadcast
from cudf_polars.utils.versions import POLARS_VERSION_LT_132

if TYPE_CHECKING:
    from typing_extensions import Self

    from polars.polars import _expr_nodes as pl_expr

    from cudf_polars.containers import DataFrame, DataType

__all__ = ["StringFunction"]

JsonDecodeType = list[tuple[str, plc.DataType, "JsonDecodeType"]]


def _dtypes_for_json_decode(dtype: DataType) -> JsonDecodeType:
    """Get the dtypes for json decode."""
    # the type checker doesn't know that this equality check implies a struct dtype.
    if dtype.id() == plc.TypeId.STRUCT:
        return [
            (field.name, child.plc_type, _dtypes_for_json_decode(child))
            for field, child in zip(
                dtype.polars_type.fields,  # type: ignore[attr-defined]
                dtype.children,
                strict=True,  # type: ignore[attr-defined]
            )
        ]
    else:
        return []


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

    _valid_ops: ClassVar[set[Name]] = {
        Name.ConcatHorizontal,
        Name.ConcatVertical,
        Name.ContainsAny,
        Name.Contains,
        Name.CountMatches,
        Name.EndsWith,
        Name.Extract,
        Name.ExtractGroups,
        Name.Find,
        Name.Head,
        Name.JsonDecode,
        Name.JsonPathMatch,
        Name.LenBytes,
        Name.LenChars,
        Name.Lowercase,
        Name.PadEnd,
        Name.PadStart,
        Name.Replace,
        Name.ReplaceMany,
        Name.Slice,
        Name.SplitN,
        Name.SplitExact,
        Name.Strptime,
        Name.StartsWith,
        Name.StripChars,
        Name.StripCharsStart,
        Name.StripCharsEnd,
        Name.StripPrefix,
        Name.StripSuffix,
        Name.Uppercase,
        Name.Reverse,
        Name.Tail,
        Name.Titlecase,
        Name.ZFill,
    }
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
        if self.name not in self._valid_ops:
            raise NotImplementedError(f"String function {self.name!r}")
        if self.name is StringFunction.Name.CountMatches:
            (literal,) = self.options
            if literal:
                raise NotImplementedError(
                    f"{literal=} is not supported for count_matches"
                )
            literal_expr = self.children[1]
            assert isinstance(literal_expr, Literal)
            pattern = literal_expr.value
            self._regex_program = self._create_regex_program(pattern)
        elif self.name is StringFunction.Name.Contains:
            literal, strict = self.options
            if not literal:
                if not strict:
                    raise NotImplementedError(
                        f"{strict=} is not supported for regex contains"
                    )
                if not isinstance(self.children[1], Literal):
                    raise NotImplementedError(
                        "Regex contains only supports a scalar pattern"
                    )
                pattern = self.children[1].value
                self._regex_program = self._create_regex_program(pattern)
        elif self.name is StringFunction.Name.Extract:
            (group_index,) = self.options
            if group_index == 0:
                raise NotImplementedError(f"{group_index=} is not supported")
            literal_expr = self.children[1]
            assert isinstance(literal_expr, Literal)
            pattern = literal_expr.value
            self._regex_program = self._create_regex_program(pattern)
        elif self.name is StringFunction.Name.ExtractGroups:
            (_, pattern) = self.options
            self._regex_program = self._create_regex_program(pattern)
        elif self.name is StringFunction.Name.Find:
            literal, strict = self.options
            if not literal:
                if not strict:
                    raise NotImplementedError(
                        f"{strict=} is not supported for regex contains"
                    )
                if not isinstance(self.children[1], Literal):
                    raise NotImplementedError(
                        "Regex contains only supports a scalar pattern"
                    )
                pattern = self.children[1].value
                self._regex_program = self._create_regex_program(pattern)
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
        elif self.name is StringFunction.Name.SplitExact:
            (_, inclusive) = self.options
            if inclusive:
                raise NotImplementedError(f"{inclusive=} is not supported for split")
        elif self.name is StringFunction.Name.Strptime:
            format, strict, exact, cache = self.options
            if not format and not strict:
                raise NotImplementedError("format inference requires strict checking")
            if cache:
                raise NotImplementedError("Strptime cache is a CPU feature")
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
        elif self.name is StringFunction.Name.ZFill:
            if isinstance(self.children[1], Literal):
                _, width = self.children
                assert isinstance(width, Literal)
                if (
                    POLARS_VERSION_LT_132
                    and width.value is not None
                    and width.value < 0
                ):  # pragma: no cover
                    dtypestr = dtype_str_repr(width.dtype.polars_type)
                    raise InvalidOperationError(
                        f"conversion from `{dtypestr}` to `u64` "
                        f"failed in column 'literal' for 1 out of "
                        f"1 values: [{width.value}]"
                    ) from None

    @staticmethod
    def _create_regex_program(
        pattern: str,
        flags: plc.strings.regex_flags.RegexFlags = plc.strings.regex_flags.RegexFlags.DEFAULT,
    ) -> plc.strings.regex_program.RegexProgram:
        if pattern == "":
            raise NotImplementedError("Empty regex pattern is not yet supported")
        try:
            return plc.strings.regex_program.RegexProgram.create(
                pattern,
                flags=flags,
            )
        except RuntimeError as e:
            raise NotImplementedError(
                f"Unsupported regex {pattern} for GPU engine."
            ) from e

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

            non_unit_sizes = [c.size for c in columns if c.size != 1]
            broadcasted = broadcast(
                *columns,
                target_length=max(non_unit_sizes) if non_unit_sizes else None,
            )

            delimiter, ignore_nulls = self.options

            return Column(
                plc.strings.combine.concatenate(
                    plc.Table([col.obj for col in broadcasted]),
                    plc.Scalar.from_py(delimiter, self.dtype.plc_type),
                    None
                    if ignore_nulls
                    else plc.Scalar.from_py(None, self.dtype.plc_type),
                    None,
                    plc.strings.combine.SeparatorOnNulls.NO,
                ),
                dtype=self.dtype,
            )
        elif self.name is StringFunction.Name.ConcatVertical:
            (child,) = self.children
            column = child.evaluate(df, context=context).astype(self.dtype)
            delimiter, ignore_nulls = self.options
            if column.null_count > 0 and not ignore_nulls:
                return Column(plc.Column.all_null_like(column.obj, 1), dtype=self.dtype)
            return Column(
                plc.strings.combine.join_strings(
                    column.obj,
                    plc.Scalar.from_py(delimiter, self.dtype.plc_type),
                    plc.Scalar.from_py(None, self.dtype.plc_type),
                ),
                dtype=self.dtype,
            )
        elif self.name is StringFunction.Name.ZFill:
            # TODO: expensive validation
            # polars pads based on bytes, libcudf by visual width
            # only pass chars if the visual width matches the byte length
            column = self.children[0].evaluate(df, context=context)
            col_len_bytes = plc.strings.attributes.count_bytes(column.obj)
            col_len_chars = plc.strings.attributes.count_characters(column.obj)
            equal = plc.binaryop.binary_operation(
                col_len_bytes,
                col_len_chars,
                plc.binaryop.BinaryOperator.NULL_EQUALS,
                plc.DataType(plc.TypeId.BOOL8),
            )
            if not plc.reduce.reduce(
                equal,
                plc.aggregation.all(),
                plc.DataType(plc.TypeId.BOOL8),
            ).to_py():
                raise InvalidOperationError(
                    "zfill only supports ascii strings with no unicode characters"
                )
            if isinstance(self.children[1], Literal):
                width = self.children[1]
                assert isinstance(width, Literal)
                if width.value is None:
                    return Column(
                        plc.Column.from_scalar(
                            plc.Scalar.from_py(None, self.dtype.plc_type),
                            column.size,
                        ),
                        self.dtype,
                    )
                return Column(
                    plc.strings.padding.zfill(column.obj, width.value), self.dtype
                )
            else:
                col_width = self.children[1].evaluate(df, context=context)
                assert isinstance(col_width, Column)
                all_gt_0 = plc.binaryop.binary_operation(
                    col_width.obj,
                    plc.Scalar.from_py(0, plc.DataType(plc.TypeId.INT64)),
                    plc.binaryop.BinaryOperator.GREATER_EQUAL,
                    plc.DataType(plc.TypeId.BOOL8),
                )

                if (
                    POLARS_VERSION_LT_132
                    and not plc.reduce.reduce(
                        all_gt_0,
                        plc.aggregation.all(),
                        plc.DataType(plc.TypeId.BOOL8),
                    ).to_py()
                ):  # pragma: no cover
                    raise InvalidOperationError("fill conversion failed.")

                return Column(
                    plc.strings.padding.zfill_by_widths(column.obj, col_width.obj),
                    self.dtype,
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
        elif self.name is StringFunction.Name.ContainsAny:
            (ascii_case_insensitive,) = self.options
            child, arg = self.children
            plc_column = child.evaluate(df, context=context).obj
            plc_targets = arg.evaluate(df, context=context).obj
            if ascii_case_insensitive:
                plc_column = plc.strings.case.to_lower(plc_column)
                plc_targets = plc.strings.case.to_lower(plc_targets)
            contains = plc.strings.find_multiple.contains_multiple(
                plc_column,
                plc_targets,
            )
            binary_or = functools.partial(
                plc.binaryop.binary_operation,
                op=plc.binaryop.BinaryOperator.BITWISE_OR,
                output_type=self.dtype.plc_type,
            )
            return Column(
                functools.reduce(binary_or, contains.columns()),
                dtype=self.dtype,
            )
        elif self.name is StringFunction.Name.CountMatches:
            (child, _) = self.children
            plc_column = child.evaluate(df, context=context).obj
            return Column(
                plc.unary.cast(
                    plc.strings.contains.count_re(plc_column, self._regex_program),
                    self.dtype.plc_type,
                ),
                dtype=self.dtype,
            )
        elif self.name is StringFunction.Name.Extract:
            (group_index,) = self.options
            plc_column = self.children[0].evaluate(df, context=context).obj
            return Column(
                plc.strings.extract.extract_single(
                    plc_column, self._regex_program, group_index - 1
                ),
                dtype=self.dtype,
            )
        elif self.name is StringFunction.Name.ExtractGroups:
            plc_column = self.children[0].evaluate(df, context=context).obj
            plc_table = plc.strings.extract.extract(
                plc_column,
                self._regex_program,
            )
            return Column(
                plc.Column.struct_from_children(plc_table.columns()),
                dtype=self.dtype,
            )
        elif self.name is StringFunction.Name.Find:
            literal, _ = self.options
            (child, expr) = self.children
            plc_column = child.evaluate(df, context=context).obj
            if literal:
                assert isinstance(expr, Literal)
                plc_column = plc.strings.find.find(
                    plc_column,
                    plc.Scalar.from_py(expr.value, expr.dtype.plc_type),
                )
            else:
                plc_column = plc.strings.findall.find_re(
                    plc_column,
                    self._regex_program,
                )
            # Polars returns None for not found, libcudf returns -1
            new_mask, null_count = plc.transform.bools_to_mask(
                plc.binaryop.binary_operation(
                    plc_column,
                    plc.Scalar.from_py(-1, plc_column.type()),
                    plc.binaryop.BinaryOperator.NOT_EQUAL,
                    plc.DataType(plc.TypeId.BOOL8),
                )
            )
            plc_column = plc.unary.cast(
                plc_column.with_mask(new_mask, null_count), self.dtype.plc_type
            )
            return Column(plc_column, dtype=self.dtype)
        elif self.name is StringFunction.Name.JsonDecode:
            plc_column = self.children[0].evaluate(df, context=context).obj
            plc_table_with_metadata = plc.io.json.read_json_from_string_column(
                plc_column,
                plc.Scalar.from_py("\n"),
                plc.Scalar.from_py("NULL"),
                _dtypes_for_json_decode(self.dtype),
            )
            return Column(
                plc.Column.struct_from_children(plc_table_with_metadata.columns),
                dtype=self.dtype,
            )
        elif self.name is StringFunction.Name.JsonPathMatch:
            (child, expr) = self.children
            plc_column = child.evaluate(df, context=context).obj
            assert isinstance(expr, Literal)
            json_path = plc.Scalar.from_py(expr.value, expr.dtype.plc_type)
            return Column(
                plc.json.get_json_object(plc_column, json_path),
                dtype=self.dtype,
            )
        elif self.name is StringFunction.Name.LenBytes:
            plc_column = self.children[0].evaluate(df, context=context).obj
            return Column(
                plc.unary.cast(
                    plc.strings.attributes.count_bytes(plc_column), self.dtype.plc_type
                ),
                dtype=self.dtype,
            )
        elif self.name is StringFunction.Name.LenChars:
            plc_column = self.children[0].evaluate(df, context=context).obj
            return Column(
                plc.unary.cast(
                    plc.strings.attributes.count_characters(plc_column),
                    self.dtype.plc_type,
                ),
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
            StringFunction.Name.SplitExact,
            StringFunction.Name.SplitN,
        }:
            is_split_n = self.name is StringFunction.Name.SplitN
            n = self.options[0]
            child, expr = self.children
            column = child.evaluate(df, context=context)
            if n == 1 and self.name is StringFunction.Name.SplitN:
                plc_column = plc.Column(
                    self.dtype.plc_type,
                    column.obj.size(),
                    None,
                    None,
                    0,
                    column.obj.offset(),
                    [column.obj],
                )
            else:
                assert isinstance(expr, Literal)
                by = plc.Scalar.from_py(expr.value, expr.dtype.plc_type)
                # See https://github.com/pola-rs/polars/issues/11640
                # for SplitN vs SplitExact edge case behaviors
                max_splits = n if is_split_n else 0
                plc_table = plc.strings.split.split.split(
                    column.obj,
                    by,
                    max_splits - 1,
                )
                children = plc_table.columns()
                ref_column = children[0]
                if (remainder := n - len(children)) > 0:
                    # Reach expected number of splits by padding with nulls
                    children.extend(
                        plc.Column.all_null_like(ref_column, ref_column.size())
                        for _ in range(remainder + int(not is_split_n))
                    )
                if not is_split_n:
                    children = children[: n + 1]
                # TODO: Use plc.Column.struct_from_children once it is generalized
                # to handle columns that don't share the same null_mask/null_count
                plc_column = plc.Column(
                    self.dtype.plc_type,
                    ref_column.size(),
                    None,
                    None,
                    0,
                    ref_column.offset(),
                    children,
                )
            return Column(plc_column, dtype=self.dtype)
        elif self.name in {
            StringFunction.Name.StripPrefix,
            StringFunction.Name.StripSuffix,
        }:
            child, expr = self.children
            plc_column = child.evaluate(df, context=context).obj
            assert isinstance(expr, Literal)
            target = plc.Scalar.from_py(expr.value, expr.dtype.plc_type)
            if self.name == StringFunction.Name.StripPrefix:
                find = plc.strings.find.starts_with
                start = len(expr.value)
                end: int | None = None
            else:
                find = plc.strings.find.ends_with
                start = 0
                end = -len(expr.value)

            mask = find(plc_column, target)
            sliced = plc.strings.slice.slice_strings(
                plc_column,
                plc.Scalar.from_py(start, plc.DataType(plc.TypeId.INT32)),
                plc.Scalar.from_py(end, plc.DataType(plc.TypeId.INT32)),
            )
            return Column(
                plc.copying.copy_if_else(
                    sliced,
                    plc_column,
                    mask,
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
                        plc.Scalar.from_py(None, self.dtype.plc_type),
                        column.size,
                    ),
                    self.dtype,
                )
            elif self.children[1].value == 0:
                result = plc.Column.from_scalar(
                    plc.Scalar.from_py("", self.dtype.plc_type),
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
                        plc.Scalar.from_py(None, self.dtype.plc_type),
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
            format, strict, _, _ = self.options
            col = self.children[0].evaluate(df, context=context)
            plc_col = col.obj
            if plc_col.null_count() == plc_col.size():
                return Column(
                    plc.Column.from_scalar(
                        plc.Scalar.from_py(None, self.dtype.plc_type),
                        plc_col.size(),
                    ),
                    self.dtype,
                )
            if format is None:
                # Polars begins inference with the first non null value
                if plc_col.null_mask() is not None:
                    boolmask = plc.unary.is_valid(plc_col)
                    table = plc.stream_compaction.apply_boolean_mask(
                        plc.Table([plc_col]), boolmask
                    )
                    filtered = table.columns()[0]
                    first_valid_data = plc.copying.get_element(filtered, 0).to_py()
                else:
                    first_valid_data = plc.copying.get_element(plc_col, 0).to_py()

                # See https://github.com/rapidsai/cudf/issues/20202 for we type ignore
                format = _infer_datetime_format(first_valid_data)  # type: ignore[arg-type]
                if not format:
                    raise InvalidOperationError(
                        "Unable to infer datetime format from data"
                    )

            is_timestamps = plc.strings.convert.convert_datetime.is_timestamp(
                plc_col, format
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
                null = plc.Scalar.from_py(None, plc_col.type())
                plc_col = plc.copying.boolean_mask_scatter(
                    [null], plc.Table([plc_col]), not_timestamps
                ).columns()[0]

            return Column(
                plc.strings.convert.convert_datetime.to_timestamps(
                    plc_col, self.dtype.plc_type, format
                ),
                dtype=self.dtype,
            )
        elif self.name is StringFunction.Name.Replace:
            col_column, col_target, col_repl = columns
            n, _ = self.options
            return Column(
                plc.strings.replace.replace(
                    col_column.obj,
                    col_target.obj_scalar,
                    col_repl.obj_scalar,
                    maxrepl=n,
                ),
                dtype=self.dtype,
            )
        elif self.name is StringFunction.Name.ReplaceMany:
            col_column, col_target, col_repl = columns
            return Column(
                plc.strings.replace.replace_multiple(
                    col_column.obj, col_target.obj, col_repl.obj
                ),
                dtype=self.dtype,
            )
        elif self.name is StringFunction.Name.PadStart:
            if POLARS_VERSION_LT_132:  # pragma: no cover
                (column,) = columns
                width, char = self.options
            else:
                (column, width_col) = columns
                (char,) = self.options
                # TODO: Maybe accept a string scalar in
                # cudf::strings::pad to avoid DtoH transfer
                # See https://github.com/rapidsai/cudf/issues/20202 for we type ignore
                width = width_col.obj.to_scalar().to_py()  # type: ignore[assignment]
            return Column(
                plc.strings.padding.pad(
                    column.obj,
                    width,
                    plc.strings.SideType.LEFT,
                    char,  # type: ignore[arg-type]
                ),
                dtype=self.dtype,
            )
        elif self.name is StringFunction.Name.PadEnd:
            if POLARS_VERSION_LT_132:  # pragma: no cover
                (column,) = columns
                width, char = self.options
            else:
                (column, width_col) = columns
                (char,) = self.options
                # TODO: Maybe accept a string scalar in
                # cudf::strings::pad to avoid DtoH transfer
                width = width_col.obj.to_scalar().to_py()  # type: ignore[assignment]
            return Column(
                plc.strings.padding.pad(
                    column.obj,
                    width,
                    plc.strings.SideType.RIGHT,
                    char,  # type: ignore[arg-type]
                ),
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


def _infer_datetime_format(val: str) -> str | None:
    # port of parts of infer.rs and patterns.rs from polars rust
    DATETIME_DMY_RE = re.compile(
        r"""
        ^
        ['"]?
        (\d{1,2})
        [-/\.]
        (?P<month>[01]?\d{1})
        [-/\.]
        (\d{4,})
        (
            [T\ ]
            (\d{1,2})
            :?
            (\d{1,2})
            (
                :?
                (\d{1,2})
                (
                    \.(\d{1,9})
                )?
            )?
        )?
        ['"]?
        $
    """,
        re.VERBOSE,
    )

    DATETIME_YMD_RE = re.compile(
        r"""
        ^
        ['"]?
        (\d{4,})
        [-/\.]
        (?P<month>[01]?\d{1})
        [-/\.]
        (\d{1,2})
        (
            [T\ ]
            (\d{1,2})
            :?
            (\d{1,2})
            (
                :?
                (\d{1,2})
                (
                    \.(\d{1,9})
                )?
            )?
        )?
        ['"]?
        $
    """,
        re.VERBOSE,
    )

    DATETIME_YMDZ_RE = re.compile(
        r"""
        ^
        ['"]?
        (\d{4,})
        [-/\.]
        (?P<month>[01]?\d{1})
        [-/\.]
        (\d{1,2})
        [T\ ]
        (\d{2})
        :?
        (\d{2})
        (
            :?
            (\d{2})
            (
                \.(\d{1,9})
            )?
        )?
        (
            [+-](\d{2})(:?(\d{2}))? | Z
        )
        ['"]?
        $
    """,
        re.VERBOSE,
    )
    PATTERN_FORMATS = {
        "DATETIME_DMY": [
            "%d-%m-%Y",
            "%d/%m/%Y",
            "%d.%m.%Y",
            "%d-%m-%Y %H:%M:%S",
            "%d/%m/%Y %H:%M:%S",
            "%d.%m.%Y %H:%M:%S",
        ],
        "DATETIME_YMD": [
            "%Y/%m/%d",
            "%Y-%m-%d",
            "%Y.%m.%d",
            "%Y-%m-%d %H:%M:%S",
            "%Y/%m/%d %H:%M:%S",
            "%Y.%m.%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S",
        ],
        "DATETIME_YMDZ": [
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%S.%f%z",
            "%Y-%m-%d %H:%M:%S%z",
        ],
    }
    for pattern_name, regex in [
        ("DATETIME_DMY", DATETIME_DMY_RE),
        ("DATETIME_YMD", DATETIME_YMD_RE),
        ("DATETIME_YMDZ", DATETIME_YMDZ_RE),
    ]:
        m = regex.match(val)
        if m:
            month = int(m.group("month"))
            if not (1 <= month <= 12):
                continue
            for fmt in PATTERN_FORMATS[pattern_name]:
                try:
                    datetime.strptime(val, fmt)
                except ValueError:  # noqa: PERF203
                    continue
                else:
                    return fmt
    return None
