# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd
import pyarrow as pa

import pylibcudf as plc

import cudf
from cudf.core.column.column import ColumnBase
from cudf.core.dtype.validators import is_dtype_obj_struct
from cudf.core.dtypes import StructDtype
from cudf.utils.dtypes import (
    dtype_from_pylibcudf_column,
    get_dtype_of_same_kind,
)
from cudf.utils.scalar import (
    maybe_nested_pa_scalar_to_py,
    pa_scalar_to_plc_scalar,
)
from cudf.utils.utils import is_na_like

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Self

    from cudf._typing import DtypeObj
    from cudf.core.column.string import StringColumn


def _maybe_na_to_none(value: Any) -> Any:
    """
    Convert NA-like values to None for pyarrow.
    """
    if is_na_like(value):
        return None
    else:
        return value


class StructColumn(ColumnBase):
    """
    Column that stores fields of values.

    Every column has n children, where n is
    the number of fields in the Struct Dtype.
    """

    _VALID_PLC_TYPES = {plc.TypeId.STRUCT}

    @classmethod
    def _validate_args(  # type: ignore[override]
        cls, plc_column: plc.Column, dtype: StructDtype
    ) -> tuple[plc.Column, StructDtype]:
        plc_column, dtype = super()._validate_args(plc_column, dtype)  # type: ignore[assignment]
        if not is_dtype_obj_struct(dtype):
            raise ValueError(f"{type(dtype).__name__} must be a StructDtype.")

        # Check field count
        if len(dtype.fields) != plc_column.num_children():
            raise ValueError(
                f"StructDtype has {len(dtype.fields)} fields, "
                f"but column has {plc_column.num_children()} children"
            )

        for i, (field_name, field_dtype) in enumerate(dtype.fields.items()):
            child = plc_column.child(i)
            try:
                ColumnBase._validate_dtype_recursively(child, field_dtype)
            except ValueError as e:
                raise ValueError(
                    f"Field '{field_name}' (index {i}) validation failed: {e}"
                ) from e

        return plc_column, dtype

    def _get_sliced_child(self, idx: int) -> ColumnBase:
        """
        Get a child column properly sliced to match the parent's view.

        Parameters
        ----------
        idx : int
            The positional index of the child column to get.

        Returns
        -------
        ColumnBase
            The child column at positional index `idx`.
        """
        if idx < 0 or idx >= self.plc_column.num_children():
            raise IndexError(
                f"Index {idx} out of range for {self.plc_column.num_children()} children"
            )

        sliced_plc_col = self.plc_column.struct_view().get_sliced_child(idx)
        sub_dtype = list(
            StructDtype.from_struct_dtype(self.dtype).fields.values()
        )[idx]
        return ColumnBase.create(
            sliced_plc_col, get_dtype_of_same_kind(self.dtype, sub_dtype)
        )

    def _prep_pandas_compat_repr(self) -> StringColumn | Self:
        """
        Preprocess Column to be compatible with pandas repr, namely handling nulls.

        * null (datetime/timedelta) = str(pd.NaT)
        * null (other types)= str(pd.NA)
        """
        # TODO: handle if self.has_nulls(): case
        return self

    def to_pandas(
        self,
        *,
        nullable: bool = False,
        arrow_type: bool = False,
    ) -> pd.Index:
        if arrow_type or isinstance(self.dtype, pd.ArrowDtype):
            return super().to_pandas(nullable=nullable, arrow_type=arrow_type)
        elif nullable and not arrow_type:
            raise NotImplementedError(
                f"pandas does not have a native nullable type for {self.dtype}."
            )
        else:
            # We cannot go via Arrow's `to_pandas` because of the following issue:
            # https://github.com/apache/arrow/issues/28428
            return pd.Index(self.to_arrow().tolist(), dtype="object")

    def element_indexing(self, index: int) -> dict[Any, Any] | None:
        result = super().element_indexing(index)
        if isinstance(result, pa.Scalar):
            py_element = maybe_nested_pa_scalar_to_py(result)
            return self.dtype._recursively_replace_fields(py_element)  # type: ignore[union-attr]
        return result

    def _cast_setitem_value(self, value: Any) -> plc.Scalar:
        if isinstance(value, dict):
            new_value = {
                field: _maybe_na_to_none(value.get(field, None))
                for field in self.dtype.fields  # type: ignore[union-attr]
            }
            return pa_scalar_to_plc_scalar(
                pa.scalar(new_value, type=self.dtype.to_arrow())  # type: ignore[union-attr]
            )
        elif value is None or value is cudf.NA:
            return pa_scalar_to_plc_scalar(
                pa.scalar(None, type=self.dtype.to_arrow())  # type: ignore[union-attr]
            )
        else:
            raise ValueError(
                f"Can not set {type(value).__name__} into StructColumn"
            )

    def copy(self, deep: bool = True) -> Self:
        # Since struct columns are immutable, both deep and
        # shallow copies share the underlying device data and mask.
        return super().copy(deep=False)

    @property
    def __cuda_array_interface__(self) -> Mapping[str, Any]:
        raise NotImplementedError(
            "Structs are not yet supported via `__cuda_array_interface__`"
        )

    def _with_type_metadata(self: StructColumn, dtype: DtypeObj) -> ColumnBase:
        # Import here to avoid circular dependency (interval imports from column)
        from cudf.core.dtypes import IntervalDtype

        # Check IntervalDtype first because it's a subclass of StructDtype
        if isinstance(dtype, IntervalDtype):
            # Import here to avoid circular dependency (interval imports from column)
            from cudf.core.column.interval import IntervalColumn

            # Determine the current subtype from the first child
            first_child_plc = self.plc_column.children()[0]
            first_child = ColumnBase.create(
                first_child_plc, dtype_from_pylibcudf_column(first_child_plc)
            )
            current_dtype = IntervalDtype(
                subtype=first_child.dtype, closed=dtype.closed
            )

            # Convert to IntervalColumn and apply target metadata
            interval_col = IntervalColumn._from_preprocessed(
                plc_column=self.plc_column,
                dtype=current_dtype,
            )
            return interval_col._with_type_metadata(dtype)
        elif isinstance(dtype, StructDtype):
            # TODO: For nested structures, stored field dtype might not reflect actual
            # child column type due to dtype metadata updates being skipped during
            # certain operations.
            new_children = tuple(
                ColumnBase.create(child, dtype_from_pylibcudf_column(child))
                for child, f in zip(
                    self.plc_column.children(),
                    dtype.fields.keys(),
                    strict=True,
                )
            )
            new_plc_column = plc.Column(
                plc.DataType(plc.TypeId.STRUCT),
                self.plc_column.size(),
                self.plc_column.data(),
                self.plc_column.null_mask(),
                self.plc_column.null_count(),
                self.plc_column.offset(),
                [child.plc_column for child in new_children],
            )
            return StructColumn._from_preprocessed(
                plc_column=new_plc_column,
                dtype=dtype,
            )
        # For pandas dtypes, store them directly in the column's dtype property
        elif isinstance(dtype, pd.ArrowDtype) and isinstance(
            dtype.pyarrow_dtype, pa.StructType
        ):
            self._dtype = dtype

        return self
