# Copyright (c) 2020-2021, NVIDIA CORPORATION.

import decimal
import pickle
from typing import Any, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
from pandas.api.extensions import ExtensionDtype
from pandas.core.arrays._arrow_utils import ArrowIntervalType

import cudf
from cudf._typing import Dtype


class CategoricalDtype(ExtensionDtype):

    ordered: Optional[bool]

    def __init__(self, categories=None, ordered: bool = None) -> None:
        """
        dtype similar to pd.CategoricalDtype with the categories
        stored on the GPU.
        """
        self._categories = self._init_categories(categories)
        self.ordered = ordered

    @property
    def categories(self) -> "cudf.core.index.Index":
        if self._categories is None:
            return cudf.core.index.as_index(
                cudf.core.column.column_empty(0, dtype="object", masked=False)
            )
        return cudf.core.index.as_index(self._categories, copy=False)

    @property
    def type(self):
        return self._categories.dtype.type

    @property
    def name(self):
        return "category"

    @property
    def str(self):
        return "|O08"

    @classmethod
    def from_pandas(cls, dtype: pd.CategoricalDtype) -> "CategoricalDtype":
        return CategoricalDtype(
            categories=dtype.categories, ordered=dtype.ordered
        )

    def to_pandas(self) -> pd.CategoricalDtype:
        if self.categories is None:
            categories = None
        else:
            categories = self.categories.to_pandas()
        return pd.CategoricalDtype(categories=categories, ordered=self.ordered)

    def _init_categories(self, categories: Any):
        if categories is None:
            return categories
        if len(categories) == 0:
            dtype = "object"  # type: Any
        else:
            dtype = None

        column = cudf.core.column.as_column(categories, dtype=dtype)

        if isinstance(column, cudf.core.column.CategoricalColumn):
            return column.categories
        else:
            return column

    def __eq__(self, other: Dtype) -> bool:
        if isinstance(other, str):
            return other == self.name
        elif other is self:
            return True
        elif not isinstance(other, self.__class__):
            return False
        elif self.ordered != other.ordered:
            return False
        elif self._categories is None or other._categories is None:
            return True
        else:
            return (
                self._categories.dtype == other._categories.dtype
                and self._categories.equals(other._categories)
            )

    def construct_from_string(self):
        raise NotImplementedError()

    def serialize(self):
        header = {}
        frames = []
        header["ordered"] = self.ordered
        if self.categories is not None:
            categories_header, categories_frames = self.categories.serialize()
        header["categories"] = categories_header
        frames.extend(categories_frames)
        return header, frames

    @classmethod
    def deserialize(cls, header, frames):
        ordered = header["ordered"]
        categories_header = header["categories"]
        categories_frames = frames
        categories_type = pickle.loads(categories_header["type-serialized"])
        categories = categories_type.deserialize(
            categories_header, categories_frames
        )
        return cls(categories=categories, ordered=ordered)


class ListDtype(ExtensionDtype):
    _typ: pa.ListType
    name: str = "list"

    def __init__(self, element_type: Any) -> None:
        if isinstance(element_type, ListDtype):
            self._typ = pa.list_(element_type._typ)
        else:
            element_type = cudf.utils.dtypes.cudf_dtype_to_pa_type(
                element_type
            )
            self._typ = pa.list_(element_type)

    @property
    def element_type(self) -> Dtype:
        if isinstance(self._typ.value_type, pa.ListType):
            return ListDtype.from_arrow(self._typ.value_type)
        else:
            return np.dtype(self._typ.value_type.to_pandas_dtype()).name

    @property
    def leaf_type(self):
        if isinstance(self.element_type, ListDtype):
            return self.element_type.leaf_type
        else:
            return self.element_type

    @property
    def type(self):
        # TODO: we should change this to return something like a
        # ListDtypeType, once we figure out what that should look like
        return pa.array

    @classmethod
    def from_arrow(cls, typ):
        obj = object.__new__(cls)
        obj._typ = typ
        return obj

    def to_arrow(self):
        return self._typ

    def __eq__(self, other):
        if isinstance(other, str):
            return other == self.name
        if not isinstance(other, ListDtype):
            return False
        return self._typ.equals(other._typ)

    def __repr__(self):
        if isinstance(self.element_type, ListDtype):
            return f"ListDtype({self.element_type.__repr__()})"
        else:
            return f"ListDtype({self.element_type})"

    def __hash__(self):
        return hash(self._typ)


class StructDtype(ExtensionDtype):

    name = "struct"

    def __init__(self, fields):
        """
        fields : dict
            A mapping of field names to dtypes
        """
        pa_fields = {
            k: cudf.utils.dtypes.cudf_dtype_to_pa_type(v)
            for k, v in fields.items()
        }
        self._typ = pa.struct(pa_fields)

    @property
    def fields(self):
        return {
            field.name: cudf.utils.dtypes.cudf_dtype_from_pa_type(field.type)
            for field in self._typ
        }

    @property
    def type(self):
        # TODO: we should change this to return something like a
        # StructDtypeType, once we figure out what that should look like
        return dict

    @classmethod
    def from_arrow(cls, typ):
        obj = object.__new__(cls)
        obj._typ = typ
        return obj

    def to_arrow(self):
        return self._typ

    def __eq__(self, other):
        if isinstance(other, str):
            return other == self.name
        if not isinstance(other, StructDtype):
            return False
        return self._typ.equals(other._typ)

    def __repr__(self):
        return f"{type(self).__name__}({self.fields})"

    def __hash__(self):
        return hash(self._typ)


class Decimal64Dtype(ExtensionDtype):

    name = "decimal"
    _metadata = ("precision", "scale")
    _MAX_PRECISION = np.floor(np.log10(np.iinfo("int64").max))

    def __init__(self, precision, scale=0):
        """
        Parameters
        ----------
        precision : int
            The total number of digits in each value of this dtype
        scale : int, optional
            The scale of the Decimal64Dtype. See Notes below.

        Notes
        -----
            When the scale is positive:
              - numbers with fractional parts (e.g., 0.0042) can be represented
              - the scale is the total number of digits to the right of the
                decimal point
            When the scale is negative:
              - only multiples of powers of 10 (including 10**0) can be
                represented (e.g., 1729, 4200, 1000000)
              - the scale represents the number of trailing zeros in the value.
            For example, 42 is representable with precision=2 and scale=0.
            13.0051 is representable with precision=6 and scale=4,
            and *not* representable with precision<6 or scale<4.
        """
        self._validate(precision, scale)
        self._typ = pa.decimal128(precision, scale)

    @property
    def precision(self):
        return self._typ.precision

    @precision.setter
    def precision(self, value):
        self._validate(value, self.scale)
        self._typ = pa.decimal128(precision=value, scale=self.scale)

    @property
    def scale(self):
        return self._typ.scale

    @property
    def type(self):
        # might need to account for precision and scale here
        return decimal.Decimal

    def to_arrow(self):
        return self._typ

    @classmethod
    def from_arrow(cls, typ):
        return cls(typ.precision, typ.scale)

    @property
    def itemsize(self):
        return 8

    def __repr__(self):
        return (
            f"{self.__class__.__name__}"
            f"(precision={self.precision}, scale={self.scale})"
        )

    def __hash__(self):
        return hash(self._typ)

    @classmethod
    def _validate(cls, precision, scale=0):
        if precision > Decimal64Dtype._MAX_PRECISION:
            raise ValueError(
                f"Cannot construct a {cls.__name__}"
                f" with precision > {cls._MAX_PRECISION}"
            )
        if abs(scale) > precision:
            raise ValueError(f"scale={scale} exceeds precision={precision}")


class IntervalDtype(StructDtype):
    name = "interval"

    def __init__(self, subtype, closed="right"):
        """
        subtype: str, np.dtype
            The dtype of the Interval bounds.
        closed: {‘right’, ‘left’, ‘both’, ‘neither’}, default ‘right’
            Whether the interval is closed on the left-side, right-side,
            both or neither. See the Notes for more detailed explanation.
        """
        super().__init__(fields={"left": subtype, "right": subtype})

        if closed in ["left", "right", "neither", "both"]:
            self.closed = closed
        else:
            raise ValueError("closed value is not valid")

    @property
    def subtype(self):
        return self.fields["left"]

    def __repr__(self):
        return f"interval[{self.fields['left']}]"

    @classmethod
    def from_arrow(cls, typ):
        return IntervalDtype(typ.subtype.to_pandas_dtype(), typ.closed)

    def to_arrow(self):

        return ArrowIntervalType(
            pa.from_numpy_dtype(self.subtype), self.closed
        )
