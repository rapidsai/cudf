# Copyright (c) 2020, NVIDIA CORPORATION.

import pickle

import numpy as np
import pandas as pd
import pyarrow as pa
from pandas.api.extensions import ExtensionDtype

import cudf
from cudf._lib.types import _Dtype
#from cudf.utils.utils import cached_property



class Generic(ExtensionDtype, _Dtype):
    pa_type = None

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return True
        if isinstance(other, Generic) and not isinstance(
            other, type(self)
        ):
            return False
        if (
            isinstance(other, self.pandas_dtype.type)
            or other is self.pandas_dtype
        ):
            return True

        if self.numpy_dtype == other:
            return True
        if isinstance(other, str) and str(self.numpy_dtype) == other:
            return True
        return False

    def __str__(self):
        return self.name

    @property
    def num(self):
        return self.numpy_dtype.num

    @property
    def numpy_dtype(self):
        return np.dtype(self.pa_type.to_pandas_dtype())

    @property
    def pandas_dtype(self):
        return pd.api.types.pandas_dtype(self.name)

    @property
    def itemsize(self):
        return self.numpy_dtype.itemsize

    @property
    def type(self):
        return CUDFType(self)

    @property
    def kind(self):
        return self.pandas_dtype.kind

    @property
    def name(self):
        return self._name

    def __repr__(self):
        return self.name

    def __hash__(self):
        return hash(self.__repr__())

    def _raise_construction_error(self):
        raise TypeError(f"Cannot create {type(self)} instances")


class Number(Generic):
    def __init__(self):
        self._raise_construction_error()


class Integer(Number):
    pass

class SignedInteger(Integer):
    pass

class UnsignedInteger(Integer):
    pass

class Inexact(Number):
    pass

class Floating(Inexact):
    pass

    @property
    def kind(self):
        return "f"


class Flexible(Generic):
    def __init__(self):
        self._construction_error()


class Datetime(Generic):
    @property
    def numpy_dtype(self):
        return {v: k for k, v in _cudf_dtype_from_numpy.items()}[self]

    @property
    def pandas_dtype(self):
        # pandas only supports nanos
        return np.dtype("datetime64[ns]")


class Timedelta(Generic):
    @property
    def numpy_dtype(self):
        return {v: k for k, v in _cudf_dtype_from_numpy.items()}[self]

    @property
    def pandas_dtype(self):
        # pandas only supports nanos
        return np.dtype("timedelta64[ns]")


class UInt8Dtype(UnsignedInteger):
    def __init__(self):
        self.pa_type = pa.uint8()
        self._name = "UInt8"


class UInt16Dtype(UnsignedInteger):
    def __init__(self):
        self.pa_type = pa.uint16()
        self._name = "UInt16"


class UInt32Dtype(UnsignedInteger):
    def __init__(self):
        self.pa_type = pa.uint32()
        self._name = "UInt32"


class UInt64Dtype(UnsignedInteger):
    def __init__(self):
        self.pa_type = pa.uint64()
        self._name = "UInt64"


class Int8Dtype(SignedInteger):
    def __init__(self):
        self.pa_type = pa.int8()
        self._name = "Int8"


class Int16Dtype(SignedInteger):
    def __init__(self):
        self.pa_type = pa.int16()
        self._name = "Int16"


class Int32Dtype(SignedInteger):
    def __init__(self):
        self.pa_type = pa.int32()
        self._name = "Int32"


class Int64Dtype(SignedInteger):
    def __init__(self):
        self.pa_type = pa.int64()
        self._name = "Int64"


class Float32Dtype(Floating):
    def __init__(self):
        self.pa_type = pa.float32()
        self._name = "Float32"


class Float64Dtype(Floating):
    def __init__(self):
        self.pa_type = pa.float64()
        self._name = "Float64"


class BooleanDtype(Generic):
    def __init__(self):
        self.pa_type = pa.bool_()
        self._name = "boolean"


class Datetime64NSDtype(Datetime):
    def __init__(self):
        self.pa_type = pa.timestamp("ns")
        self._name = "Datetime64NS"
        self._time_unit = "ns"


class Datetime64USDtype(Datetime):
    def __init__(self):
        self.pa_type = pa.timestamp("us")
        self._name = "Datetime64US"
        self._time_unit = "us"


class Datetime64MSDtype(Datetime):
    def __init__(self):
        self.pa_type = pa.timestamp("ms")
        self._name = "Datetime64MS"
        self._time_unit = "ms"


class Datetime64SDtype(Datetime):
    def __init__(self):
        self.pa_type = pa.timestamp("s")
        self._name = "Datetime64S"
        self._time_unit = "s"


class Timedelta64NSDtype(Timedelta):
    def __init__(self):
        self.pa_type = pa.duration("ns")
        self._name = "Timedelta64NS"
        self._time_unit = "ns"


class Timedelta64USDtype(Timedelta):
    def __init__(self):
        self.pa_type = pa.duration("us")
        self._name = "Timedelta64US"
        self._time_unit = "us"


class Timedelta64MSDtype(Timedelta):
    def __init__(self):
        self.pa_type = pa.duration("ms")
        self._name = "Timedelta64MS"
        self._time_unit = "ms"


class Timedelta64SDtype(Timedelta):
    def __init__(self):
        self.pa_type = pa.duration("s")
        self._name = "Timedelta64S"
        self._time_unit = "s"


class StringDtype(Flexible):
    def __init__(self):
        self.pa_type = pa.string()
        self._name = "string"


class CUDFType(object):
    def __init__(self, parent_dtype):
        self.parent_dtype = parent_dtype

    def __call__(self, arg):
        return cudf.Scalar(arg, dtype=self.parent_dtype)


def cudf_dtype_from_string(obj):
    if obj == "category":
        return CategoricalDtype()
    try:
        np_dtype = np.dtype(obj)
        return cudf_dtype_from_numpy(np_dtype)
    except TypeError:
        result = _cudf_dtype_from_string.get(obj, None)
        if not result:
            import pdb
            pdb.set_trace()
            raise TypeError(f"Could not find a cuDF dtype matching {obj}")
        return result


def cudf_dtype_from_numpy(obj):
    if obj is np.str_:
        return StringDtype()
    elif obj is np.number:
        return cudf.Number
    elif obj in {np.datetime64, np.dtype('datetime64')}:
        return cudf.Datetime
    elif obj in {np.timedelta64, np.dtype('timedelta64')}:
        return cudf.Timedelta
    dtype = np.dtype(obj)
    if dtype.type is np.str_:
        return StringDtype()
    result = _cudf_dtype_from_numpy.get(dtype, None)
    if not result:
        raise TypeError(f"Could not find a cuDF dtype matching {obj}")
    return result


def cudf_dtype_from_pandas(obj):
    if isinstance(obj, pd.core.arrays.numpy_.PandasDtype):
        try:
            return cudf_dtype_from_numpy(obj.numpy_dtype)
        except TypeError:
            result = _cudf_dtype_from_pandas.get(obj, None)
            if not result:
                raise TypeError(f"Could not find a cuDF dtype matching {obj}")
            return result


def dtype(obj):
    if isinstance(obj, Generic):
        return obj
    elif type(obj) is type and issubclass(obj, Generic):
        if obj in cant_construct_dtypes:
            return obj
        else:
            return obj()
    elif isinstance(obj, np.dtype) or (
        isinstance(obj, type) and issubclass(obj, (np.generic, np.dtype))
    ):
        return cudf_dtype_from_numpy(obj)
    elif isinstance(obj, str):
        return cudf_dtype_from_string(obj)
    if isinstance(obj, pd.CategoricalDtype):
        return cudf.CategoricalDtype.from_pandas(obj)
    elif isinstance(obj, (ExtensionDtype, pd.core.arrays.numpy_.PandasDtype)):
        return cudf_dtype_from_pandas(obj)
    elif isinstance(obj, pa.lib.DataType):
        return cudf_dtype_from_pyarrow[obj]
    elif obj is str:
        return cudf.StringDtype()
    elif obj is int:
        return cudf.Int64Dtype()
    elif obj is float:
        return cudf.Float64Dtype()
    elif obj is None:
        return None
    elif obj is np.object:
        return 
    else:
        raise TypeError(f"Could not find cuDF dtype matching {obj}")

class CategoricalDtype(Generic):
    def __init__(self, categories=None, ordered=None):
        """
        dtype similar to pd.CategoricalDtype with the categories
        stored on the GPU.
        """
        self._categories = self._init_categories(categories)
        self.ordered = ordered

    def __repr__(self):
        return self.to_pandas().__repr__()

    @property
    def pandas_dtype(self):
        return self.to_pandas()

    def __hash__(self):
        return hash(self.__repr__())

    @property
    def categories(self):
        if self._categories is None:
            return cudf.core.index.as_index(
                cudf.core.column.column_empty(0, dtype="object", masked=False)
            )
        return cudf.core.index.as_index(self._categories)

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
    def from_pandas(cls, dtype):
        return CategoricalDtype(
            categories=dtype.categories, ordered=dtype.ordered
        )

    def to_pandas(self):
        if self.categories is None:
            categories = None
        else:
            categories = self.categories.to_pandas()
        return pd.CategoricalDtype(categories=categories, ordered=self.ordered)

    def _init_categories(self, categories):
        if categories is None:
            return categories
        if len(categories) == 0:
            dtype = "object"
        else:
            dtype = None

        column = cudf.core.column.as_column(categories, dtype=dtype)

        if isinstance(column, cudf.core.column.CategoricalColumn):
            return column.categories
        else:
            return column

    def __eq__(self, other):
        if isinstance(other, str):
            return other == self.name
        elif other is self:
            return True
        elif not isinstance(other, type(self)):
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

    @property
    def kind(self):
        return "O"


class ListDtype(Generic):

    name = "list"

    def __init__(self, element_type):
        if isinstance(element_type, ListDtype):
            self._typ = pa.list_(element_type._typ)
        else:
            element_type = cudf.utils.dtypes.np_to_pa_dtype(
                cudf.dtype(element_type)
            )
            self._typ = pa.list_(element_type)

    @property
    def element_type(self):
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
    def kind(self):
        return "O"

    @property
    def type(self):
        # TODO: we should change this to return something like a
        # ListDtypeType, once we figure out what that should look like
        return pa.array

    @classmethod
    def from_arrow(cls, typ):
        obj = ListDtype.__new__(ListDtype)
        obj._typ = typ
        return obj

    def to_arrow(self):
        return self._typ

    def to_pandas(self):
        super().to_pandas(integer_object_nulls=True)

    def __eq__(self, other):
        if isinstance(other, str):
            return other == self.name
        if type(other) is not ListDtype:
            return False
        return self._typ.equals(other._typ)

    def __repr__(self):
        if isinstance(self.element_type, ListDtype):
            return f"ListDtype({self.element_type.__repr__()})"
        else:
            return f"ListDtype({self.element_type})"


cudf_dtype_from_pyarrow = {
    pa.uint8(): UInt8Dtype(),
    pa.uint16(): UInt16Dtype(),
    pa.uint32(): UInt32Dtype(),
    pa.uint64(): UInt64Dtype(),
    pa.int8(): Int8Dtype(),
    pa.int16(): Int16Dtype(),
    pa.int32(): Int32Dtype(),
    pa.int64(): Int64Dtype(),
    pa.bool_(): BooleanDtype(),
    pa.string(): StringDtype(),
    pa.float32(): Float32Dtype(),
    pa.float64(): Float64Dtype(),
    pa.timestamp("ns"): Datetime64NSDtype(),
    pa.timestamp("us"): Datetime64USDtype(),
    pa.timestamp("ms"): Datetime64MSDtype(),
    pa.timestamp("s"): Datetime64SDtype(),
    pa.duration("ns"): Timedelta64NSDtype(),
    pa.duration("us"): Timedelta64USDtype(),
    pa.duration("ms"): Timedelta64MSDtype(),
    pa.duration("s"): Timedelta64SDtype(),
    pa.date32(): Datetime64NSDtype(),
    pa.null(): None,
}

_cudf_dtype_from_numpy = {
    np.dtype("int8"): Int8Dtype(),
    np.dtype("int16"): Int16Dtype(),
    np.dtype("int32"): Int32Dtype(),
    np.dtype("int64"): Int64Dtype(),
    np.dtype("uint8"): UInt8Dtype(),
    np.dtype("uint16"): UInt16Dtype(),
    np.dtype("uint32"): UInt32Dtype(),
    np.dtype("uint64"): UInt64Dtype(),
    np.dtype("bool"): BooleanDtype(),
    np.dtype("U"): StringDtype(),
    np.dtype("object"): StringDtype(),
    np.dtype("float32"): Float32Dtype(),
    np.dtype("float64"): Float64Dtype(),
    np.dtype("datetime64[ns]"): Datetime64NSDtype(),
    np.dtype("datetime64[us]"): Datetime64USDtype(),
    np.dtype("datetime64[ms]"): Datetime64MSDtype(),
    np.dtype("datetime64[s]"): Datetime64SDtype(),
    np.dtype("timedelta64[ns]"): Timedelta64NSDtype(),
    np.dtype("timedelta64[us]"): Timedelta64USDtype(),
    np.dtype("timedelta64[ms]"): Timedelta64MSDtype(),
    np.dtype("timedelta64[s]"): Timedelta64SDtype(),
}

_cudf_dtype_from_string = {
    "UInt8": UInt8Dtype,
    "UInt16": UInt16Dtype,
    "UInt32": UInt32Dtype,
    "UInt64": UInt64Dtype,
    "Int8": Int8Dtype,
    "Int16": Int16Dtype,
    "Int32": Int32Dtype,
    "Int64": Int64Dtype,
    "Float": Float64Dtype,
    "Float32": Float32Dtype,
    "Float64": Float64Dtype,
    "Boolean": BooleanDtype,
    "String": StringDtype,
    "Datetime64NS": Datetime64NSDtype,
    "Datetime64US": Datetime64USDtype,
    "Datetime64MS": Datetime64MSDtype,
    "Datetime64S": Datetime64SDtype,
    "Timedelta64NS": Timedelta64NSDtype,
    "Timedelta64US": Timedelta64USDtype,
    "Timedelta64MS": Timedelta64MSDtype,
    "Timedelta64S": Timedelta64SDtype,
}

_cudf_dtype_from_pandas = {
    pd.UInt8Dtype(): UInt8Dtype(),
    pd.UInt16Dtype(): UInt16Dtype(),
    pd.UInt32Dtype(): UInt32Dtype(),
    pd.UInt64Dtype(): UInt64Dtype(),
    pd.Int8Dtype(): Int8Dtype(),
    pd.Int16Dtype(): Int16Dtype(),
    pd.Int32Dtype(): Int32Dtype(),
    pd.Int64Dtype(): Int64Dtype(),
    pd.StringDtype(): StringDtype(),
    pd.BooleanDtype(): BooleanDtype(),
}

cant_construct_dtypes = {Number, Integer, UnsignedInteger, Floating, Inexact, Timedelta}
