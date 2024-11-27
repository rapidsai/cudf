# Copyright (c) 2020-2024, NVIDIA CORPORATION.
from __future__ import annotations

import decimal
import operator
import pickle
import textwrap
import warnings
from functools import cached_property
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pyarrow as pa
from pandas.api import types as pd_types
from pandas.api.extensions import ExtensionDtype
from pandas.core.arrays.arrow.extension_types import ArrowIntervalType

import cudf
from cudf.core._compat import PANDAS_GE_210, PANDAS_LT_300
from cudf.core.abc import Serializable
from cudf.utils.docutils import doc_apply

if PANDAS_GE_210:
    PANDAS_NUMPY_DTYPE = pd.core.dtypes.dtypes.NumpyEADtype
else:
    PANDAS_NUMPY_DTYPE = pd.core.dtypes.dtypes.PandasDtype

if TYPE_CHECKING:
    from collections.abc import Callable

    from cudf._typing import Dtype
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
    #  first, check if `arbitrary` is one of our extension types:
    if isinstance(arbitrary, cudf.core.dtypes._BaseDtype):
        return arbitrary

    # next, try interpreting arbitrary as a NumPy dtype that we support:
    try:
        np_dtype = np.dtype(arbitrary)
    except TypeError:
        pass
    else:
        if np_dtype.kind in set("OU"):
            return np.dtype("object")
        elif np_dtype not in cudf._lib.types.SUPPORTED_NUMPY_TO_LIBCUDF_TYPES:
            raise TypeError(f"Unsupported type {np_dtype}")
        return np_dtype

    if isinstance(arbitrary, str) and arbitrary in {"hex", "hex32", "hex64"}:
        # read_csv only accepts "hex"
        # e.g. test_csv_reader_hexadecimals, test_csv_reader_hexadecimal_overflow
        return arbitrary

    # use `pandas_dtype` to try and interpret
    # `arbitrary` as a Pandas extension type.
    #  Return the corresponding NumPy/cuDF type.
    pd_dtype = pd.api.types.pandas_dtype(arbitrary)
    if cudf.api.types._is_pandas_nullable_extension_dtype(pd_dtype):
        if cudf.get_option("mode.pandas_compatible"):
            raise NotImplementedError(
                "Nullable types not supported in pandas compatibility mode"
            )
        elif isinstance(pd_dtype, pd.StringDtype):
            return np.dtype("object")
        else:
            return dtype(pd_dtype.numpy_dtype)
    elif isinstance(pd_dtype, PANDAS_NUMPY_DTYPE):
        return dtype(pd_dtype.numpy_dtype)
    elif isinstance(pd_dtype, pd.CategoricalDtype):
        return cudf.CategoricalDtype.from_pandas(pd_dtype)
    elif isinstance(pd_dtype, pd.IntervalDtype):
        return cudf.IntervalDtype.from_pandas(pd_dtype)
    elif isinstance(pd_dtype, pd.DatetimeTZDtype):
        return pd_dtype
    else:
        raise TypeError(f"Cannot interpret {arbitrary} as a valid cuDF dtype")


def _decode_type(
    cls: type,
    header: dict,
    frames: list,
    is_valid_class: Callable[[type, type], bool] = operator.is_,
) -> tuple[dict, list, type]:
    """Decode metadata-encoded type and check validity

    Parameters
    ----------
    cls : type
        class performing deserialization
    header : dict
        metadata for deserialization
    frames : list
        buffers containing data for deserialization
    is_valid_class : Callable
        function to call to check if the encoded class type is valid for
        serialization by `cls` (default is to check type equality), called
        as `is_valid_class(decoded_class, cls)`.

    Returns
    -------
    tuple
        Tuple of validated headers, frames, and the decoded class
        constructor.

    Raises
    ------
    AssertionError
        if the number of frames doesn't match the count encoded in the
        headers, or `is_valid_class` is not true.
    """
    assert header["frame_count"] == len(frames), (
        f"Deserialization expected {header['frame_count']} frames, "
        f"but received {len(frames)}."
    )
    klass = pickle.loads(header["type-serialized"])
    assert is_valid_class(
        klass, cls
    ), f"Header-encoded {klass=} does not match decoding {cls=}."
    return header, frames, klass


class _BaseDtype(ExtensionDtype, Serializable):
    # Base type for all cudf-specific dtypes
    pass


class CategoricalDtype(_BaseDtype):
    """
    Type for categorical data with the categories and orderedness.

    Parameters
    ----------
    categories : sequence, optional
        Must be unique, and must not contain any nulls.
        The categories are stored in an Index,
        and if an index is provided the dtype of that index will be used.
    ordered : bool or None, default False
        Whether or not this categorical is treated as a ordered categorical.
        None can be used to maintain the ordered value of existing categoricals
        when used in operations that combine categoricals, e.g. astype, and
        will resolve to False if there is no existing ordered to maintain.

    Attributes
    ----------
    categories
    ordered

    Methods
    -------
    from_pandas
    to_pandas

    Examples
    --------
    >>> import cudf
    >>> dtype = cudf.CategoricalDtype(categories=['b', 'a'], ordered=True)
    >>> cudf.Series(['a', 'b', 'a', 'c'], dtype=dtype)
    0       a
    1       b
    2       a
    3    <NA>
    dtype: category
    Categories (2, object): ['b' < 'a']
    """

    def __init__(self, categories=None, ordered: bool = False) -> None:
        self._categories = self._init_categories(categories)
        self._ordered = ordered

    @property
    def categories(self) -> cudf.Index:
        """
        An ``Index`` containing the unique categories allowed.

        Examples
        --------
        >>> import cudf
        >>> dtype = cudf.CategoricalDtype(categories=['b', 'a'], ordered=True)
        >>> dtype.categories
        Index(['b', 'a'], dtype='object')
        """
        if self._categories is None:
            col = cudf.core.column.column_empty(
                0, dtype="object", masked=False
            )
        else:
            col = self._categories
        return cudf.Index._from_column(col)

    @property
    def type(self):
        return self._categories.dtype.type

    @property
    def name(self):
        return "category"

    @property
    def str(self):
        return "|O08"

    @property
    def ordered(self) -> bool:
        """
        Whether the categories have an ordered relationship.
        """
        return self._ordered

    @classmethod
    def from_pandas(cls, dtype: pd.CategoricalDtype) -> "CategoricalDtype":
        """
        Convert a ``pandas.CategrocialDtype`` to ``cudf.CategoricalDtype``

        Examples
        --------
        >>> import cudf
        >>> import pandas as pd
        >>> pd_dtype = pd.CategoricalDtype(categories=['b', 'a'], ordered=True)
        >>> pd_dtype
        CategoricalDtype(categories=['b', 'a'], ordered=True, categories_dtype=object)
        >>> cudf_dtype = cudf.CategoricalDtype.from_pandas(pd_dtype)
        >>> cudf_dtype
        CategoricalDtype(categories=['b', 'a'], ordered=True, categories_dtype=object)
        """
        return CategoricalDtype(
            categories=dtype.categories, ordered=dtype.ordered
        )

    def to_pandas(self) -> pd.CategoricalDtype:
        """
        Convert a ``cudf.CategoricalDtype`` to ``pandas.CategoricalDtype``

        Examples
        --------
        >>> import cudf
        >>> dtype = cudf.CategoricalDtype(categories=['b', 'a'], ordered=True)
        >>> dtype
        CategoricalDtype(categories=['b', 'a'], ordered=True, categories_dtype=object)
        >>> dtype.to_pandas()
        CategoricalDtype(categories=['b', 'a'], ordered=True, categories_dtype=object)
        """
        if self._categories is None:
            categories = None
        elif self._categories.dtype.kind == "f":
            categories = self._categories.dropna().to_pandas()
        else:
            categories = self._categories.to_pandas()
        return pd.CategoricalDtype(categories=categories, ordered=self.ordered)

    def _init_categories(
        self, categories: Any
    ) -> cudf.core.column.ColumnBase | None:
        if categories is None:
            return categories
        if len(categories) == 0 and not isinstance(
            getattr(categories, "dtype", None),
            (cudf.IntervalDtype, pd.IntervalDtype),
        ):
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
        header["frame_count"] = len(frames)
        return header, frames

    @classmethod
    def deserialize(cls, header, frames):
        header, frames, klass = _decode_type(cls, header, frames)
        ordered = header["ordered"]
        categories_header = header["categories"]
        categories_frames = frames
        categories_type = pickle.loads(categories_header["type-serialized"])
        categories = categories_type.deserialize(
            categories_header, categories_frames
        )
        return klass(categories=categories, ordered=ordered)

    def __repr__(self):
        return self.to_pandas().__repr__()


class ListDtype(_BaseDtype):
    """
    Type to represent list data.

    Parameters
    ----------
    element_type : object
        A dtype with which represents the element types in the list.

    Attributes
    ----------
    element_type
    leaf_type

    Methods
    -------
    from_arrow
    to_arrow

    Examples
    --------
    >>> import cudf
    >>> list_dtype = cudf.ListDtype("int32")
    >>> list_dtype
    ListDtype(int32)

    A nested list dtype can be created by:

    >>> nested_list_dtype = cudf.ListDtype(list_dtype)
    >>> nested_list_dtype
    ListDtype(ListDtype(int32))
    """

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

    @cached_property
    def element_type(self) -> Dtype:
        """
        Returns the element type of the ``ListDtype``.

        Returns
        -------
        Dtype

        Examples
        --------
        >>> import cudf
        >>> deep_nested_type = cudf.ListDtype(cudf.ListDtype(cudf.ListDtype("float32")))
        >>> deep_nested_type
        ListDtype(ListDtype(ListDtype(float32)))
        >>> deep_nested_type.element_type
        ListDtype(ListDtype(float32))
        >>> deep_nested_type.element_type.element_type
        ListDtype(float32)
        >>> deep_nested_type.element_type.element_type.element_type
        'float32'
        """
        if isinstance(self._typ.value_type, pa.ListType):
            return ListDtype.from_arrow(self._typ.value_type)
        elif isinstance(self._typ.value_type, pa.StructType):
            return StructDtype.from_arrow(self._typ.value_type)
        else:
            return cudf.dtype(self._typ.value_type.to_pandas_dtype())

    @cached_property
    def leaf_type(self):
        """
        Returns the type of the leaf values.

        Examples
        --------
        >>> import cudf
        >>> deep_nested_type = cudf.ListDtype(cudf.ListDtype(cudf.ListDtype("float32")))
        >>> deep_nested_type
        ListDtype(ListDtype(ListDtype(float32)))
        >>> deep_nested_type.leaf_type
        'float32'
        """
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
        """
        Creates a ``ListDtype`` from ``pyarrow.ListType``.

        Parameters
        ----------
        typ : pyarrow.ListType
            A ``pyarrow.ListType`` that has to be converted to
            ``ListDtype``.

        Returns
        -------
        obj : ``ListDtype``

        Examples
        --------
        >>> import cudf
        >>> import pyarrow as pa
        >>> arrow_type = pa.infer_type([[1]])
        >>> arrow_type
        ListType(list<item: int64>)
        >>> list_dtype = cudf.ListDtype.from_arrow(arrow_type)
        >>> list_dtype
        ListDtype(int64)
        """
        obj = object.__new__(cls)
        obj._typ = typ
        return obj

    def to_arrow(self):
        """
        Convert to a ``pyarrow.ListType``

        Examples
        --------
        >>> import cudf
        >>> list_dtype = cudf.ListDtype(cudf.ListDtype("float32"))
        >>> list_dtype
        ListDtype(ListDtype(float32))
        >>> list_dtype.to_arrow()
        ListType(list<item: list<item: float>>)
        """
        return self._typ

    def __eq__(self, other):
        if isinstance(other, str):
            return other == self.name
        if not isinstance(other, ListDtype):
            return False
        return self._typ.equals(other._typ)

    def __repr__(self):
        if isinstance(self.element_type, (ListDtype, StructDtype)):
            return f"{type(self).__name__}({self.element_type!r})"
        else:
            return f"{type(self).__name__}({self.element_type})"

    def __hash__(self):
        return hash(self._typ)

    def serialize(self) -> tuple[dict, list]:
        header: dict[str, Dtype] = {}
        header["type-serialized"] = pickle.dumps(type(self))

        frames = []

        if isinstance(self.element_type, _BaseDtype):
            header["element-type"], frames = self.element_type.serialize()
        else:
            header["element-type"] = getattr(
                self.element_type, "name", self.element_type
            )
        header["frame_count"] = len(frames)
        return header, frames

    @classmethod
    def deserialize(cls, header: dict, frames: list):
        header, frames, klass = _decode_type(cls, header, frames)
        if isinstance(header["element-type"], dict):
            element_type = pickle.loads(
                header["element-type"]["type-serialized"]
            ).deserialize(header["element-type"], frames)
        else:
            element_type = header["element-type"]
        return klass(element_type=element_type)

    @cached_property
    def itemsize(self):
        return self.element_type.itemsize


class StructDtype(_BaseDtype):
    """
    Type to represent a struct data.

    Parameters
    ----------
    fields : dict
        A mapping of field names to dtypes, the dtypes can themselves
        be of ``StructDtype`` too.

    Attributes
    ----------
    fields
    itemsize

    Methods
    -------
    from_arrow
    to_arrow

    Examples
    --------
    >>> import cudf
    >>> struct_dtype = cudf.StructDtype({"a": "int64", "b": "string"})
    >>> struct_dtype
    StructDtype({'a': dtype('int64'), 'b': dtype('O')})

    A nested ``StructDtype`` can also be constructed in the following way:

    >>> nested_struct_dtype = cudf.StructDtype({"dict_data": struct_dtype, "c": "uint8"})
    >>> nested_struct_dtype
    StructDtype({'dict_data': StructDtype({'a': dtype('int64'), 'b': dtype('O')}), 'c': dtype('uint8')})
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
        """
        Returns an ordered dict of column name and dtype key-value.

        Examples
        --------
        >>> import cudf
        >>> struct_dtype = cudf.StructDtype({"a": "int64", "b": "string"})
        >>> struct_dtype
        StructDtype({'a': dtype('int64'), 'b': dtype('O')})
        >>> struct_dtype.fields
        {'a': dtype('int64'), 'b': dtype('O')}
        """
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
        """
        Convert a ``pyarrow.StructType`` to ``StructDtype``.

        Examples
        --------
        >>> import cudf
        >>> import pyarrow as pa
        >>> pa_struct_type = pa.struct({'x': pa.int32(), 'y': pa.string()})
        >>> pa_struct_type
        StructType(struct<x: int32, y: string>)
        >>> cudf.StructDtype.from_arrow(pa_struct_type)
        StructDtype({'x': dtype('int32'), 'y': dtype('O')})
        """
        obj = object.__new__(cls)
        obj._typ = typ
        return obj

    def to_arrow(self):
        """
        Convert a ``StructDtype`` to a ``pyarrow.StructType``.

        Examples
        --------
        >>> import cudf
        >>> struct_type = cudf.StructDtype({"x": "int32", "y": "string"})
        >>> struct_type
        StructDtype({'x': dtype('int32'), 'y': dtype('O')})
        >>> struct_type.to_arrow()
        StructType(struct<x: int32, y: string>)
        """
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

    def serialize(self) -> tuple[dict, list]:
        header: dict[str, Any] = {}
        header["type-serialized"] = pickle.dumps(type(self))

        frames: list[Buffer] = []

        fields: dict[str, bytes | tuple[Any, tuple[int, int]]] = {}

        for k, dtype in self.fields.items():
            if isinstance(dtype, _BaseDtype):
                dtype_header, dtype_frames = dtype.serialize()
                fields[k] = (
                    dtype_header,
                    (len(frames), len(frames) + len(dtype_frames)),
                )
                frames.extend(dtype_frames)
            else:
                fields[k] = pickle.dumps(dtype)
        header["fields"] = fields
        header["frame_count"] = len(frames)
        return header, frames

    @classmethod
    def deserialize(cls, header: dict, frames: list):
        header, frames, klass = _decode_type(cls, header, frames)
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
                fields[k] = pickle.loads(dtype)
        return cls(fields)

    @cached_property
    def itemsize(self):
        return sum(
            cudf.utils.dtypes.cudf_dtype_from_pa_type(field.type).itemsize
            for field in self._typ
        )


decimal_dtype_template = textwrap.dedent(
    """
        Type to represent a ``decimal{size}`` data.

        Parameters
        ----------
        precision : int
            The total number of digits in each value of this dtype
        scale : int, optional
            The scale of the dtype. See Notes below.

        Attributes
        ----------
        precision
        scale
        itemsize

        Methods
        -------
        to_arrow
        from_arrow

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

        Examples
        --------
        >>> import cudf
        >>> decimal{size}_dtype = cudf.Decimal{size}Dtype(precision=9, scale=2)
        >>> decimal{size}_dtype
        Decimal{size}Dtype(precision=9, scale=2)
    """
)


class DecimalDtype(_BaseDtype):
    _metadata = ("precision", "scale")

    def __init__(self, precision, scale=0):
        self._validate(precision, scale)
        self._typ = pa.decimal128(precision, scale)

    @property
    def str(self):
        return f"{self.name!s}({self.precision}, {self.scale})"

    @property
    def precision(self):
        """
        The decimal precision, in number of decimal digits (an integer).
        """
        return self._typ.precision

    @precision.setter
    def precision(self, value):
        self._validate(value, self.scale)
        self._typ = pa.decimal128(precision=value, scale=self.scale)

    @property
    def scale(self):
        """
        The decimal scale (an integer).
        """
        return self._typ.scale

    @property
    def itemsize(self):
        """
        Length of one column element in bytes.
        """
        return self.ITEMSIZE

    @property
    def type(self):
        # might need to account for precision and scale here
        return decimal.Decimal

    def to_arrow(self):
        """
        Return the equivalent ``pyarrow`` dtype.
        """
        return self._typ

    @classmethod
    def from_arrow(cls, typ):
        """
        Construct a cudf decimal dtype from a ``pyarrow`` dtype

        Examples
        --------
        >>> import cudf
        >>> import pyarrow as pa
        >>> pa_type = pa.decimal128(precision=9, scale=2)

        Constructing a ``Decimal32Dtype``:

        >>> cudf.Decimal32Dtype.from_arrow(pa_type)
        Decimal64Dtype(precision=9, scale=2)

        Constructing a ``Decimal64Dtype``:

        >>> cudf.Decimal64Dtype.from_arrow(pa_type)
        Decimal64Dtype(precision=9, scale=2)

        Constructing a ``Decimal128Dtype``:

        >>> cudf.Decimal128Dtype.from_arrow(pa_type)
        Decimal128Dtype(precision=9, scale=2)
        """
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
        Create a cudf.DecimalDtype from a decimal.Decimal object
        """
        metadata = decimal.as_tuple()
        precision = max(len(metadata.digits), -metadata.exponent)
        return cls(precision, -metadata.exponent)

    def serialize(self) -> tuple[dict, list]:
        return (
            {
                "type-serialized": pickle.dumps(type(self)),
                "precision": self.precision,
                "scale": self.scale,
                "frame_count": 0,
            },
            [],
        )

    @classmethod
    def deserialize(cls, header: dict, frames: list):
        header, frames, klass = _decode_type(
            cls, header, frames, is_valid_class=issubclass
        )
        klass = pickle.loads(header["type-serialized"])
        return klass(header["precision"], header["scale"])

    def __eq__(self, other: Dtype) -> bool:
        if other is self:
            return True
        elif not isinstance(other, self.__class__):
            return False
        return self.precision == other.precision and self.scale == other.scale

    def __hash__(self):
        return hash(self._typ)


@doc_apply(
    decimal_dtype_template.format(
        size="32",
    )
)
class Decimal32Dtype(DecimalDtype):
    name = "decimal32"
    MAX_PRECISION = np.floor(np.log10(np.iinfo("int32").max))
    ITEMSIZE = 4


@doc_apply(
    decimal_dtype_template.format(
        size="64",
    )
)
class Decimal64Dtype(DecimalDtype):
    name = "decimal64"
    MAX_PRECISION = np.floor(np.log10(np.iinfo("int64").max))
    ITEMSIZE = 8


@doc_apply(
    decimal_dtype_template.format(
        size="128",
    )
)
class Decimal128Dtype(DecimalDtype):
    name = "decimal128"
    MAX_PRECISION = 38
    ITEMSIZE = 16


class IntervalDtype(StructDtype):
    """
    subtype: str, np.dtype
        The dtype of the Interval bounds.
    closed: {'right', 'left', 'both', 'neither'}, default 'right'
        Whether the interval is closed on the left-side, right-side,
        both or neither. See the Notes for more detailed explanation.
    """

    name = "interval"

    def __init__(self, subtype, closed="right"):
        super().__init__(fields={"left": subtype, "right": subtype})

        if closed is None:
            closed = "right"
        if closed in ["left", "right", "neither", "both"]:
            self.closed = closed
        else:
            raise ValueError("closed value is not valid")

    @property
    def subtype(self):
        return self.fields["left"]

    def __repr__(self) -> str:
        return f"interval[{self.subtype}, {self.closed}]"

    def __str__(self) -> str:
        return self.__repr__()

    @classmethod
    def from_arrow(cls, typ):
        return IntervalDtype(typ.subtype.to_pandas_dtype(), typ.closed)

    def to_arrow(self):
        return ArrowIntervalType(
            pa.from_numpy_dtype(self.subtype), self.closed
        )

    @classmethod
    def from_pandas(cls, pd_dtype: pd.IntervalDtype) -> "IntervalDtype":
        return cls(subtype=pd_dtype.subtype, closed=pd_dtype.closed)

    def to_pandas(self) -> pd.IntervalDtype:
        return pd.IntervalDtype(subtype=self.subtype, closed=self.closed)

    def __eq__(self, other):
        if isinstance(other, str):
            # This means equality isn't transitive but mimics pandas
            return other in (self.name, str(self))
        return (
            type(self) is type(other)
            and self.subtype == other.subtype
            and self.closed == other.closed
        )

    def __hash__(self):
        return hash((self.subtype, self.closed))

    def serialize(self) -> tuple[dict, list]:
        header = {
            "type-serialized": pickle.dumps(type(self)),
            "fields": pickle.dumps((self.subtype, self.closed)),
            "frame_count": 0,
        }
        return header, []

    @classmethod
    def deserialize(cls, header: dict, frames: list):
        header, frames, klass = _decode_type(cls, header, frames)
        klass = pickle.loads(header["type-serialized"])
        subtype, closed = pickle.loads(header["fields"])
        return klass(subtype, closed=closed)


def _is_categorical_dtype(obj):
    if obj is None:
        return False

    if isinstance(
        obj,
        (
            pd.CategoricalDtype,
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
            pd.CategoricalDtype,
            pd.CategoricalDtype.type,
        )
    ):
        return True
    if isinstance(obj, (np.ndarray, np.dtype)):
        return False
    if isinstance(obj, str) and obj == "category":
        return True
    if isinstance(obj, cudf.core.index.BaseIndex):
        return obj._is_categorical()
    if isinstance(
        obj,
        (
            cudf.Series,
            cudf.core.column.ColumnBase,
            pd.Index,
            pd.Series,
        ),
    ):
        try:
            return isinstance(cudf.dtype(obj.dtype), cudf.CategoricalDtype)
        except TypeError:
            return False
    if hasattr(obj, "type"):
        if obj.type is pd.CategoricalDtype.type:
            return True
    # TODO: A lot of the above checks are probably redundant and should be
    # farmed out to this function here instead.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pd_types.is_categorical_dtype(obj)


def is_categorical_dtype(obj):
    """Check whether an array-like or dtype is of the Categorical dtype.

    .. deprecated:: 24.04
       Use isinstance(dtype, cudf.CategoricalDtype) instead

    Parameters
    ----------
    obj : array-like or dtype
        The array-like or dtype to check.

    Returns
    -------
    bool
        Whether or not the array-like or dtype is of a categorical dtype.
    """
    # Do not remove until pandas 3.0 support is added.
    assert PANDAS_LT_300, "Need to drop after pandas-3.0 support is added."
    warnings.warn(
        "is_categorical_dtype is deprecated and will be removed in a future "
        "version. Use isinstance(dtype, cudf.CategoricalDtype) instead",
        DeprecationWarning,
    )
    return _is_categorical_dtype(obj)


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
        or (
            hasattr(obj, "dtype")
            and isinstance(obj.dtype, cudf.core.dtypes.ListDtype)
        )
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
        or (
            hasattr(obj, "dtype")
            and isinstance(obj.dtype, cudf.core.dtypes.StructDtype)
        )
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


def _is_interval_dtype(obj):
    return (
        isinstance(
            obj,
            (
                cudf.core.dtypes.IntervalDtype,
                pd.IntervalDtype,
            ),
        )
        or obj is cudf.core.dtypes.IntervalDtype
        or (isinstance(obj, cudf.core.index.BaseIndex) and obj._is_interval())
        or (
            isinstance(obj, str) and obj == cudf.core.dtypes.IntervalDtype.name
        )
        or (
            isinstance(
                getattr(obj, "dtype", None),
                (pd.IntervalDtype, cudf.core.dtypes.IntervalDtype),
            )
        )
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
    warnings.warn(
        "is_interval_dtype is deprecated and will be removed in a "
        "future version. Use `isinstance(dtype, cudf.IntervalDtype)` instead",
        DeprecationWarning,
    )
    return _is_interval_dtype(obj)


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
