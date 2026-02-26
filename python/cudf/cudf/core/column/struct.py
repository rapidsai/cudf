# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd
import pyarrow as pa

import cudf
from cudf.core.column.column import ColumnBase
from cudf.core.dtypes import StructDtype
from cudf.utils.dtypes import get_dtype_of_same_kind
from cudf.utils.scalar import pa_scalar_to_plc_scalar
from cudf.utils.utils import is_na_like

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Self

    import pylibcudf as plc

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
