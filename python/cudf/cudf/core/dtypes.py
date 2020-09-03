# Copyright (c) 2020, NVIDIA CORPORATION.

import pickle

import numpy as np
import pandas as pd
import pyarrow as pa
from pandas.api.extensions import ExtensionDtype
from cudf._lib.types import _Dtype
import cudf

pa_to_pd_dtypes = {
    pa.uint8(): pd.UInt8Dtype(),
    pa.uint16(): pd.UInt16Dtype(),
    pa.uint32(): pd.UInt32Dtype(),
    pa.uint64(): pd.UInt64Dtype(),
    pa.int8(): pd.Int8Dtype(),
    pa.int16(): pd.Int16Dtype(),
    pa.int32(): pd.Int32Dtype(),
    pa.int64(): pd.Int64Dtype(),
    pa.bool_(): pd.BooleanDtype(),
    pa.string(): pd.StringDtype(),
    pa.float32(): np.float32(),
    pa.float64(): np.float64(),
    pa.timestamp("ns"): np.dtype("datetime64[ns]"),
    pa.timestamp("us"): np.dtype("datetime64[us]"),
    pa.timestamp("ms"): np.dtype("datetime64[ms]"),
    pa.timestamp("s"): np.dtype("datetime64[s]"),
    pa.duration("ns"): np.dtype('timedelta64[ns]'),
    pa.duration("us"): np.dtype('timedelta64[us]'),
    pa.duration("ms"): np.dtype('timedelta64[ms]'),
    pa.duration("s"): np.dtype('timedelta64[s]'),
}

pa_to_np_dtypes = {
    pa.uint8(): np.dtype("uint8"),
    pa.uint16(): np.dtype("uint16"),
    pa.uint32(): np.dtype("uint32"),
    pa.uint64(): np.dtype("uint64"),
    pa.int8(): np.dtype("int8"),
    pa.int16(): np.dtype("int16"),
    pa.int32(): np.dtype("int32"),
    pa.int64(): np.dtype("int64"),
    pa.bool_(): np.dtype("bool"),
    pa.string(): np.dtype("object"),
    pa.float32(): np.dtype("float32"),
    pa.float64(): np.dtype("float64"),
    pa.timestamp("ns"): np.dtype("datetime64[ns]"),
    pa.timestamp("us"): np.dtype("datetime64[us]"),
    pa.timestamp("ms"): np.dtype("datetime64[ms]"),
    pa.timestamp("s"): np.dtype("datetime64[s]"),
    pa.duration("ns"): np.dtype('timedelta64[ns]'),
    pa.duration("us"): np.dtype('timedelta64[us]'),
    pa.duration("ms"): np.dtype('timedelta64[ms]'),
    pa.duration("s"): np.dtype('timedelta64[s]'),
    None: None,
}


class Generic(ExtensionDtype, _Dtype):
    pa_type = None

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return True
        if isinstance(other, Generic) and not isinstance(other, self.__class__):
            return False
        if (
            isinstance(other, self.to_pandas.__class__)
            or other is self.to_pandas.__class__
        ):
            return True

        if self.to_numpy == other:
            return True
        if isinstance(other, str) and str(self.to_numpy) == other:
            return True
        return False

    def __str__(self):
        return self.name

    @property
    def num(self):
        return self.to_numpy.num

    @property
    def to_numpy(self):
        return pa_to_np_dtypes[self.pa_type]

    @property
    def to_pandas(self):
        return pa_to_pd_dtypes[self.pa_type]

    @property
    def itemsize(self):
        return self.to_numpy.itemsize

    @property
    def type(self):
        if isinstance(self, (Floating, Datetime)):
            return self.to_numpy.type
        else:
            return self.to_pandas.type

    @property
    def kind(self):
        if isinstance(self, Floating):
            return "f"
        else:
            return self.to_pandas.kind

    @property
    def name(self):
        return self._name

    def __repr__(self):
        return self.pa_type.__repr__()

    def __hash__(self):
        return hash(self.__repr__())

class Number(Generic):
    pass

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

class Flexible(Generic):
    pass

class Datetime(Generic):    
    pass

class Timedelta(Generic):
    pass

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
        self._name = "Boolean"

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
        self.pa_type = pa.duration('ns')
        self._name = "Timedelta64NS"
        self._time_unit = 'ns'

class Timedelta64USDtype(Timedelta):
    def __init__(self):
        self.pa_type = pa.duration('us')
        self._name = "Timedelta64US"
        self._time_unit = 'us'

class Timedelta64MSDtype(Timedelta):
    def __init__(self):
        self.pa_type = pa.duration('ms')
        self._name = "Timedelta64MS"
        self._time_unit = 'ms'

class Timedelta64SDtype(Timedelta):
    def __init__(self):
        self.pa_type = pa.duration('s')
        self._name = "Timedelta64S"
        self._time_unit = 's'

class StringDtype(Flexible):
    is_string = True

    def __init__(self):
        self.pa_type = pa.string()
        self._name = "String"


def make_dtype_from_string(obj):
    if obj in {"str", "string", "object", "O"}:
        return StringDtype()
    elif "datetime" in obj or "Datetime" in obj:
        if obj in {"datetime64[ns]", 'Datetime64NS'}:
            return Datetime64NSDtype()
        elif obj in {"datetime64[us]", "Datetime64US"}:
            return Datetime64USDtype()
        elif obj in {"datetime64[ms]", "Datetime64MS"}:
            return Datetime64MSDtype()
        elif obj in {"datetime64[s]", "Datetime64S"}:
            return Datetime64SDtype()
    elif "int" in obj or "Int" in obj:
        if obj in {"int", "Int", "int64", "Int64"}:
            return Int64Dtype()
        elif obj in {"int32", "Int32"}:
            return Int32Dtype()
        elif obj in {"int16", "Int16"}:
            return Int16Dtype()
        elif obj in {"int8", "Int8"}:
            return Int8Dtype()
        elif obj in {"uint64", "UInt64"}:
            return UInt64Dtype()
        elif obj in {"uint32", "UInt32"}:
            return UInt32Dtype()
        elif obj in {"uint16", "UInt16"}:
            return UInt16Dtype()
        elif obj in {"uint8", "UInt8"}:
            return UInt8Dtype()
    elif "float" in obj or "Float" in obj:
        if obj in {"float64", "Float64", 'float', 'Float'}:
            return Float64Dtype()
        elif obj in {"float32", "Float32"}:
            return Float32Dtype()
    elif "bool" in obj:
        return BooleanDtype()
    elif "category" in obj:
        return "category"
    elif "timedelta" in obj:
        if obj in {'timedelta64[ns]', "Timedelta64NS"}:
            return Timedelta64NSDtype()
        if obj in {'timedelta64[us]', "Timedelta64US"}:
            return Timedelta64USDtype()
        if obj in {'timedelta64[ms]', "Timedelta64MS"}:
            return Timedelta64MSDtype()
        if obj in {'timedelta64[s]', "Timedelta64S"}:
            return Timedelta64SDtype()
    else:
        try:
            return np_to_cudf_dtypes[np.dtype(obj)]
        except:
            return None
def make_dtype_from_numpy(obj):
    np_to_pd_types = {v: k for k, v in pd_to_np_dtypes.items()}
    result = np_to_pd_types.get(obj)
    return result


def dtype(obj):

    if obj is None:
        return None
    if obj is str:
        return cudf.StringDtype()
    if obj is int:
        return cudf.Int64Dtype()
    if obj is float:
        return cudf.Float64Dtype()
    if isinstance(obj, pd.CategoricalDtype):
        return cudf.CategoricalDtype.from_pandas(obj)
    if isinstance(obj, CategoricalDtype):
        if obj is 'category':
            return cudf.CategoricalDtype()
        return obj
    elif isinstance(obj, Generic):
        return obj
    elif issubclass(obj.__class__, Generic):
        return obj()
    if isinstance(obj, np.dtype):
        if obj.type is np.str_:
            return StringDtype()
        else:
            return np_to_cudf_dtypes.get(obj, None)
    elif isinstance(obj, pa.lib.DataType):
        return pa_to_cudf_dtypes[obj]
    elif isinstance(obj, str):
        return make_dtype_from_string(obj)
    elif obj in pd_to_cudf_dtypes.keys():
        return pd_to_cudf_dtypes[obj]
    elif isinstance(obj, pd.core.arrays.numpy_.PandasDtype):
        return make_dtype_from_string(obj.name)
    elif obj is np.number:
        return cudf.Number
    elif obj is np.datetime64:
        return cudf.Datetime
    elif obj is np.timedelta64:
        return cudf.Timedelta


    else:
        try:
            if issubclass(obj, np.generic):
                return np_to_cudf_dtypes[np.dtype(obj)]
        except:
            import pdb
            pdb.set_trace()
    


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

    @property
    def kind(self):
        return 'O'


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
        return 'O'

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


pa_to_cudf_dtypes = {
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
    pa.null(): None
}

np_to_cudf_dtypes = {
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

pd_to_cudf_dtypes = {
    pd.Int8Dtype(): Int8Dtype(),
    pd.Int16Dtype(): Int16Dtype(),
    pd.Int32Dtype(): Int32Dtype(),
    pd.Int64Dtype(): Int64Dtype(),
    pd.UInt8Dtype(): UInt8Dtype(),
    pd.UInt16Dtype(): UInt16Dtype(),
    pd.UInt32Dtype(): UInt32Dtype(),
    pd.UInt64Dtype(): UInt64Dtype(),
    pd.BooleanDtype(): BooleanDtype(),
    pd.StringDtype(): StringDtype(),
}
