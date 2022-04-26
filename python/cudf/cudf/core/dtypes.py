# Copyright (c) 2020-2022, NVIDIA CORPORATION.

import decimal
import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
from pandas.api import types as pd_types
from pandas.api.extensions import ExtensionDtype
from pandas.core.arrays._arrow_utils import ArrowIntervalType
from pandas.core.dtypes.dtypes import (
    CategoricalDtype as pd_CategoricalDtype,
    CategoricalDtypeType as pd_CategoricalDtypeType,
)

import cudf
from cudf._typing import Dtype
from cudf.core.abc import Serializable
from cudf.core.buffer import Buffer


def dtype(arbitrary):
    """
    Return the cuDF-supported dtype corresponding to `arbitrary`.

    Parameters
    ----------
    arbitrary: dtype or scalar-like

    Returns
    -------
    dtype: the cuDF-supported dtype that best matches `arbitrary`
    """
    # first, try interpreting arbitrary as a NumPy dtype that we support:
    try:
        np_dtype = np.dtype(arbitrary)
        if np_dtype.kind in ("OU"):
            return np.dtype("object")
    except TypeError:
        pass
    else:
        if np_dtype not in cudf._lib.types.SUPPORTED_NUMPY_TO_LIBCUDF_TYPES:
            raise TypeError(f"Unsupported type {np_dtype}")
        return np_dtype

    #  next, check if `arbitrary` is one of our extension types:
    if isinstance(arbitrary, cudf.core.dtypes._BaseDtype):
        return arbitrary

    # use `pandas_dtype` to try and interpret
    # `arbitrary` as a Pandas extension type.
    #  Return the corresponding NumPy/cuDF type.
    pd_dtype = pd.api.types.pandas_dtype(arbitrary)
    try:
        return dtype(pd_dtype.numpy_dtype)
    except AttributeError:
        if isinstance(pd_dtype, pd.CategoricalDtype):
            return cudf.CategoricalDtype.from_pandas(pd_dtype)
        elif isinstance(pd_dtype, pd.StringDtype):
            return np.dtype("object")
        elif isinstance(pd_dtype, pd.IntervalDtype):
            return cudf.IntervalDtype.from_pandas(pd_dtype)
        else:
            raise TypeError(
                f"Cannot interpret {arbitrary} as a valid cuDF dtype"
            )


class _BaseDtype(ExtensionDtype, Serializable):
    # Base type for all cudf-specific dtypes
    pass


class CategoricalDtype(_BaseDtype):
    """
    dtype similar to pd.CategoricalDtype with the categories
    stored on the GPU.
    """

    ordered: Optional[bool]

    def __init__(self, categories=None, ordered: bool = None) -> None:
        self._categories = self._init_categories(categories)
        self.ordered = ordered

    @property
    def categories(self) -> "cudf.core.index.BaseIndex":
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
        if self._categories is None:
            categories = None
        else:
            if isinstance(
                self._categories, (cudf.Float32Index, cudf.Float64Index)
            ):
                categories = self._categories.dropna().to_pandas()
            else:
                categories = self._categories.to_pandas()
        return pd.CategoricalDtype(categories=categories, ordered=self.ordered)

    def _init_categories(self, categories: Any):
        if categories is None:
            return categories
        if len(categories) == 0 and not is_interval_dtype(categories):
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
        header["type-serialized"] = pickle.dumps(type(self))
        header["ordered"] = self.ordered

        frames = []

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


class ListDtype(_BaseDtype):
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
        elif isinstance(self._typ.value_type, pa.StructType):
            return StructDtype.from_arrow(self._typ.value_type)
        else:
            return cudf.dtype(self._typ.value_type.to_pandas_dtype()).name

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
        if isinstance(self.element_type, (ListDtype, StructDtype)):
            return f"{type(self).__name__}({self.element_type.__repr__()})"
        else:
            return f"{type(self).__name__}({self.element_type})"

    def __hash__(self):
        return hash(self._typ)

    def serialize(self) -> Tuple[dict, list]:
        header: Dict[str, Dtype] = {}
        header["type-serialized"] = pickle.dumps(type(self))

        frames = []

        if isinstance(self.element_type, _BaseDtype):
            header["element-type"], frames = self.element_type.serialize()
        else:
            header["element-type"] = self.element_type

        return header, frames

    @classmethod
    def deserialize(cls, header: dict, frames: list):
        if isinstance(header["element-type"], dict):
            element_type = pickle.loads(
                header["element-type"]["type-serialized"]
            ).deserialize(header["element-type"], frames)
        else:
            element_type = header["element-type"]

        return cls(element_type=element_type)


class StructDtype(_BaseDtype):
    """
    fields : dict
        A mapping of field names to dtypes
    """

    name = "struct"

    def __init__(self, fields):
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

    def serialize(self) -> Tuple[dict, list]:
        header: Dict[str, Any] = {}
        header["type-serialized"] = pickle.dumps(type(self))

        frames: List[Buffer] = []

        fields = {}

        for k, dtype in self.fields.items():
            if isinstance(dtype, _BaseDtype):
                dtype_header, dtype_frames = dtype.serialize()
                fields[k] = (
                    dtype_header,
                    (len(frames), len(frames) + len(dtype_frames)),
                )
                frames.extend(dtype_frames)
            else:
                fields[k] = dtype
        header["fields"] = fields

        return header, frames

    @classmethod
    def deserialize(cls, header: dict, frames: list):
        fields = {}
        for k, dtype in header["fields"].items():
            if isinstance(dtype, tuple):
                dtype_header, (start, stop) = dtype
                fields[k] = pickle.loads(
                    dtype_header["type-serialized"]
                ).deserialize(
                    dtype_header,
                    frames[start:stop],
                )
            else:
                fields[k] = dtype
        return cls(fields)


class DecimalDtype(_BaseDtype):
    """
    Parameters
    ----------
    precision : int
        The total number of digits in each value of this dtype
    scale : int, optional
        The scale of the dtype. See Notes below.

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

    _metadata = ("precision", "scale")

    def __init__(self, precision, scale=0):
        self._validate(precision, scale)
        self._typ = pa.decimal128(precision, scale)

    @property
    def str(self):
        return f"{str(self.name)}({self.precision}, {self.scale})"

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
    def itemsize(self):
        return self.ITEMSIZE

    @property
    def type(self):
        # might need to account for precision and scale here
        return decimal.Decimal

    def to_arrow(self):
        return self._typ

    @classmethod
    def from_arrow(cls, typ):
        return cls(typ.precision, typ.scale)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}"
            f"(precision={self.precision}, scale={self.scale})"
        )

    @classmethod
    def _validate(cls, precision, scale=0):
        if precision > cls.MAX_PRECISION:
            raise ValueError(
                f"Cannot construct a {cls.__name__}"
                f" with precision > {cls.MAX_PRECISION}"
            )
        if abs(scale) > precision:
            raise ValueError(f"scale={scale} exceeds precision={precision}")

    @classmethod
    def _from_decimal(cls, decimal):
        """
        Create a cudf.Decimal32Dtype from a decimal.Decimal object
        """
        metadata = decimal.as_tuple()
        precision = max(len(metadata.digits), -metadata.exponent)
        return cls(precision, -metadata.exponent)

    def serialize(self) -> Tuple[dict, list]:
        return (
            {
                "type-serialized": pickle.dumps(type(self)),
                "precision": self.precision,
                "scale": self.scale,
            },
            [],
        )

    @classmethod
    def deserialize(cls, header: dict, frames: list):
        return cls(header["precision"], header["scale"])

    def __eq__(self, other: Dtype) -> bool:
        if other is self:
            return True
        elif not isinstance(other, self.__class__):
            return False
        return self.precision == other.precision and self.scale == other.scale

    def __hash__(self):
        return hash(self._typ)


class Decimal32Dtype(DecimalDtype):
    name = "decimal32"
    MAX_PRECISION = np.floor(np.log10(np.iinfo("int32").max))
    ITEMSIZE = 4


class Decimal64Dtype(DecimalDtype):
    name = "decimal64"
    MAX_PRECISION = np.floor(np.log10(np.iinfo("int64").max))
    ITEMSIZE = 8


class Decimal128Dtype(DecimalDtype):
    name = "decimal128"
    MAX_PRECISION = 38
    ITEMSIZE = 16


class IntervalDtype(StructDtype):
    """
    subtype: str, np.dtype
        The dtype of the Interval bounds.
    closed: {‘right’, ‘left’, ‘both’, ‘neither’}, default ‘right’
        Whether the interval is closed on the left-side, right-side,
        both or neither. See the Notes for more detailed explanation.
    """

    name = "interval"

    def __init__(self, subtype, closed="right"):
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

    @classmethod
    def from_pandas(cls, pd_dtype: pd.IntervalDtype) -> "IntervalDtype":
        return cls(
            subtype=pd_dtype.subtype
        )  # TODO: needs `closed` when we upgrade Pandas


def is_categorical_dtype(obj):
    """Check whether an array-like or dtype is of the Categorical dtype.

    Parameters
    ----------
    obj : array-like or dtype
        The array-like or dtype to check.

    Returns
    -------
    bool
        Whether or not the array-like or dtype is of a categorical dtype.
    """
    if obj is None:
        return False

    if isinstance(
        obj,
        (
            pd_CategoricalDtype,
            cudf.CategoricalDtype,
            cudf.core.index.CategoricalIndex,
            cudf.core.column.CategoricalColumn,
            pd.Categorical,
            pd.CategoricalIndex,
        ),
    ):
        return True
    # Note that we cannot directly use `obj in (...)`  because that triggers
    # equality as well as identity checks and pandas extension dtypes won't
    # allow converting that equality check to a boolean; `__nonzero__` is
    # disabled because they treat dtypes as "array-like".
    if any(
        obj is t
        for t in (
            cudf.CategoricalDtype,
            pd_CategoricalDtype,
            pd_CategoricalDtypeType,
        )
    ):
        return True
    if isinstance(obj, (np.ndarray, np.dtype)):
        return False
    if isinstance(obj, str) and obj == "category":
        return True
    if isinstance(
        obj,
        (
            cudf.Index,
            cudf.Series,
            cudf.core.column.ColumnBase,
            pd.Index,
            pd.Series,
        ),
    ):
        return is_categorical_dtype(obj.dtype)
    if hasattr(obj, "type"):
        if obj.type is pd_CategoricalDtypeType:
            return True
    # TODO: A lot of the above checks are probably redundant and should be
    # farmed out to this function here instead.
    return pd_types.is_categorical_dtype(obj)


def is_list_dtype(obj):
    """Check whether an array-like or dtype is of the list dtype.

    Parameters
    ----------
    obj : array-like or dtype
        The array-like or dtype to check.

    Returns
    -------
    bool
        Whether or not the array-like or dtype is of the list dtype.
    """
    return (
        type(obj) is cudf.core.dtypes.ListDtype
        or obj is cudf.core.dtypes.ListDtype
        or type(obj) is cudf.core.column.ListColumn
        or obj is cudf.core.column.ListColumn
        or (isinstance(obj, str) and obj == cudf.core.dtypes.ListDtype.name)
        or (hasattr(obj, "dtype") and is_list_dtype(obj.dtype))
    )


def is_struct_dtype(obj):
    """Check whether an array-like or dtype is of the struct dtype.

    Parameters
    ----------
    obj : array-like or dtype
        The array-like or dtype to check.

    Returns
    -------
    bool
        Whether or not the array-like or dtype is of the struct dtype.
    """
    # TODO: This behavior is currently inconsistent for interval types. the
    # actual class IntervalDtype will return False, but instances (e.g.
    # IntervalDtype(int)) will return True. For now this is not being changed
    # since the interval dtype is being modified as part of the array refactor,
    # but this behavior should be made consistent afterwards.
    return (
        isinstance(obj, cudf.core.dtypes.StructDtype)
        or obj is cudf.core.dtypes.StructDtype
        or (isinstance(obj, str) and obj == cudf.core.dtypes.StructDtype.name)
        or (hasattr(obj, "dtype") and is_struct_dtype(obj.dtype))
    )


def is_decimal_dtype(obj):
    """Check whether an array-like or dtype is of the decimal dtype.

    Parameters
    ----------
    obj : array-like or dtype
        The array-like or dtype to check.

    Returns
    -------
    bool
        Whether or not the array-like or dtype is of the decimal dtype.
    """
    return (
        is_decimal32_dtype(obj)
        or is_decimal64_dtype(obj)
        or is_decimal128_dtype(obj)
    )


def is_interval_dtype(obj):
    """Check whether an array-like or dtype is of the interval dtype.

    Parameters
    ----------
    obj : array-like or dtype
        The array-like or dtype to check.

    Returns
    -------
    bool
        Whether or not the array-like or dtype is of the interval dtype.
    """
    # TODO: Should there be any branch in this function that calls
    # pd.api.types.is_interval_dtype?
    return (
        isinstance(
            obj,
            (
                cudf.core.dtypes.IntervalDtype,
                pd.core.dtypes.dtypes.IntervalDtype,
            ),
        )
        or obj is cudf.core.dtypes.IntervalDtype
        or (
            isinstance(obj, str) and obj == cudf.core.dtypes.IntervalDtype.name
        )
        or (hasattr(obj, "dtype") and is_interval_dtype(obj.dtype))
    )


def is_decimal32_dtype(obj):
    return (
        type(obj) is cudf.core.dtypes.Decimal32Dtype
        or obj is cudf.core.dtypes.Decimal32Dtype
        or (
            isinstance(obj, str)
            and obj == cudf.core.dtypes.Decimal32Dtype.name
        )
        or (hasattr(obj, "dtype") and is_decimal32_dtype(obj.dtype))
    )


def is_decimal64_dtype(obj):
    return (
        type(obj) is cudf.core.dtypes.Decimal64Dtype
        or obj is cudf.core.dtypes.Decimal64Dtype
        or (
            isinstance(obj, str)
            and obj == cudf.core.dtypes.Decimal64Dtype.name
        )
        or (hasattr(obj, "dtype") and is_decimal64_dtype(obj.dtype))
    )


def is_decimal128_dtype(obj):
    return (
        type(obj) is cudf.core.dtypes.Decimal128Dtype
        or obj is cudf.core.dtypes.Decimal128Dtype
        or (
            isinstance(obj, str)
            and obj == cudf.core.dtypes.Decimal128Dtype.name
        )
        or (hasattr(obj, "dtype") and is_decimal128_dtype(obj.dtype))
    )
