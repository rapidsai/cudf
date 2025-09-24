# Copyright (c) 2020-2025, NVIDIA CORPORATION.
from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Any

import pandas as pd
import pyarrow as pa

import cudf
from cudf.core.column.column import ColumnBase
from cudf.core.dtypes import StructDtype
from cudf.utils.dtypes import (
    is_dtype_obj_struct,
    pyarrow_dtype_to_cudf_dtype,
)
from cudf.utils.scalar import (
    maybe_nested_pa_scalar_to_py,
    pa_scalar_to_plc_scalar,
)
from cudf.utils.utils import _is_null_host_scalar

if TYPE_CHECKING:
    from typing_extensions import Self

    import pylibcudf as plc

    from cudf._typing import Dtype
    from cudf.core.buffer import Buffer
    from cudf.core.column.string import StringColumn


def _maybe_na_to_none(value: Any) -> Any:
    """
    Convert NA-like values to None for pyarrow.
    """
    if _is_null_host_scalar(value):
        return None
    else:
        return value


class StructColumn(ColumnBase):
    """
    Column that stores fields of values.

    Every column has n children, where n is
    the number of fields in the Struct Dtype.
    """

    def __init__(
        self,
        data: None,
        size: int,
        dtype: StructDtype,
        mask: Buffer | None = None,
        offset: int = 0,
        null_count: int | None = None,
        children: tuple[ColumnBase, ...] = (),
    ):
        if data is not None:
            raise ValueError("data must be None.")
        dtype = self._validate_dtype_instance(dtype)
        super().__init__(
            data=data,
            size=size,
            dtype=dtype,
            mask=mask,
            offset=offset,
            null_count=null_count,
            children=children,
        )

    def _prep_pandas_compat_repr(self) -> StringColumn | Self:
        """
        Preprocess Column to be compatible with pandas repr, namely handling nulls.

        * null (datetime/timedelta) = str(pd.NaT)
        * null (other types)= str(pd.NA)
        """
        # TODO: handle if self.has_nulls(): case
        return self

    @staticmethod
    def _validate_dtype_instance(dtype: StructDtype) -> StructDtype:
        # IntervalDtype is a subclass of StructDtype, so compare types exactly
        if (
            not cudf.get_option("mode.pandas_compatible")
            and type(dtype) is not StructDtype
        ) or (
            cudf.get_option("mode.pandas_compatible")
            and not is_dtype_obj_struct(dtype)
        ):
            raise ValueError(
                f"{type(dtype).__name__} must be a StructDtype exactly."
            )
        return dtype

    @property
    def base_size(self):
        if self.base_children:
            return len(self.base_children[0])
        else:
            return self.size + self.offset

    def to_arrow(self) -> pa.Array:
        children = [child.to_arrow() for child in self.children]
        dtype = (
            pyarrow_dtype_to_cudf_dtype(self.dtype)
            if isinstance(self.dtype, pd.ArrowDtype)
            else self.dtype
        )
        pa_type = pa.struct(
            {
                field: child.type
                for field, child in zip(dtype.fields, children, strict=True)
            }
        )

        if self.mask is not None:
            buffers = (pa.py_buffer(self.mask.memoryview()),)
        else:
            buffers = (None,)

        return pa.StructArray.from_buffers(
            pa_type, len(self), buffers, children=children
        )

    def to_pandas(
        self,
        *,
        nullable: bool = False,
        arrow_type: bool = False,
    ) -> pd.Index:
        # We cannot go via Arrow's `to_pandas` because of the following issue:
        # https://issues.apache.org/jira/browse/ARROW-12680
        if (
            arrow_type
            or nullable
            or (
                cudf.get_option("mode.pandas_compatible")
                and isinstance(self.dtype, pd.ArrowDtype)
            )
        ):
            return super().to_pandas(nullable=nullable, arrow_type=arrow_type)
        else:
            return pd.Index(self.to_arrow().tolist(), dtype="object")

    @cached_property
    def memory_usage(self) -> int:
        n = super().memory_usage
        for child in self.children:
            n += child.memory_usage

        return n

    def element_indexing(self, index: int) -> dict:
        result = super().element_indexing(index)
        if isinstance(result, pa.Scalar):
            py_element = maybe_nested_pa_scalar_to_py(result)
            return self.dtype._recursively_replace_fields(py_element)
        return result

    def _cast_setitem_value(self, value: Any) -> plc.Scalar:
        if isinstance(value, dict):
            new_value = {
                field: _maybe_na_to_none(value.get(field, None))
                for field in self.dtype.fields
            }
            return pa_scalar_to_plc_scalar(
                pa.scalar(new_value, type=self.dtype.to_arrow())
            )
        elif value is None or value is cudf.NA:
            return pa_scalar_to_plc_scalar(
                pa.scalar(None, type=self.dtype.to_arrow())
            )
        else:
            raise ValueError(
                f"Can not set {type(value).__name__} into StructColumn"
            )

    def copy(self, deep: bool = True) -> Self:
        # Since struct columns are immutable, both deep and
        # shallow copies share the underlying device data and mask.
        result = super().copy(deep=False)
        if deep:
            result = result._rename_fields(self.dtype.fields.keys())
        return result

    def _rename_fields(self, names) -> Self:
        """
        Return a StructColumn with the same field values as this StructColumn,
        but with the field names equal to `names`.
        """
        dtype = StructDtype(
            {
                name: col.dtype
                for name, col in zip(names, self.children, strict=True)
            }
        )
        return self._with_type_metadata(dtype)  # type: ignore[return-value]

    @property
    def __cuda_array_interface__(self):
        raise NotImplementedError(
            "Structs are not yet supported via `__cuda_array_interface__`"
        )

    def _with_type_metadata(self: StructColumn, dtype: Dtype) -> StructColumn:
        from cudf.core.column import IntervalColumn
        from cudf.core.dtypes import IntervalDtype

        # Check IntervalDtype first because it's a subclass of StructDtype
        if isinstance(dtype, IntervalDtype):
            return IntervalColumn(
                data=None,
                size=self.size,
                dtype=dtype,
                mask=self.base_mask,
                offset=self.offset,
                null_count=self.null_count,
                children=self.base_children,  # type: ignore[arg-type]
            )
        elif isinstance(dtype, StructDtype):
            return StructColumn(
                data=None,
                dtype=dtype,
                children=tuple(
                    self.base_children[i]._with_type_metadata(dtype.fields[f])
                    for i, f in enumerate(dtype.fields.keys())
                ),
                mask=self.base_mask,
                size=self.size,
                offset=self.offset,
                null_count=self.null_count,
            )
        # For pandas dtypes, store them directly in the column's dtype property
        elif isinstance(dtype, pd.ArrowDtype) and isinstance(
            dtype.pyarrow_dtype, pa.StructType
        ):
            self._dtype = dtype

        return self
