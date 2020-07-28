# Copyright (c) 2020, NVIDIA CORPORATION.

import pickle

import numpy as np
import pandas as pd
import pyarrow as pa
from pandas.api.extensions import ExtensionDtype

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
    pa.timestamp('ns'): np.dtype('datetime64[ns]'),
    pa.timestamp('us'): np.dtype('datetime64[us]'),
    pa.timestamp('ms'): np.dtype('datetime64[ms]'),
    pa.timestamp('s'): np.dtype('datetime64[s]'),
}

pa_to_np_dtypes = {
    pa.uint8(): np.dtype('uint8'),
    pa.uint16(): np.dtype('uint16'),
    pa.uint32(): np.dtype('uint32'),
    pa.uint64(): np.dtype('uint64'),
    pa.int8(): np.dtype('int8'),
    pa.int16(): np.dtype('int16'),
    pa.int32(): np.dtype('int32'),
    pa.int64(): np.dtype('int64'),
    pa.bool_(): np.dtype('bool'),
    pa.string(): np.dtype('object'),
    pa.float32(): np.dtype('float32'),
    pa.float64(): np.dtype('float64'),
    pa.timestamp('ns'): np.dtype('datetime64[ns]'),
    pa.timestamp('us'): np.dtype('datetime64[us]'),
    pa.timestamp('ms'): np.dtype('datetime64[ms]'),
    pa.timestamp('s'): np.dtype('datetime64[s]'),
}

class Dtype(ExtensionDtype):
    def __init__(self, arg):
        cudf_dtype = make_dtype_from_obj(arg)
        cudf_dtype.__init__(self)

    def __eq__(self, other):
        if isinstance(other, self.to_pandas.__class__) or other is self.to_pandas.__class__:
            return True
        if self.to_numpy == other:
            return True
        raise NotImplementedError

    @property
    def to_numpy(self):
        return pa_to_np_dtypes[self.pa_type]

    @property
    def to_pandas(self):
        return pa_to_pd_dtypes[self.pa_type]

    @property
    def type(self):
        return self.pandas_dtype().type

class UInt8Dtype(Dtype):
    def __init__(self):
        self.pa_type = pa.uint8()
        
class UInt16Dtype(Dtype):
    def __init__(self):
        self.pa_type = pa.uint16()

class UInt32Dtype(Dtype):
    def __init__(self):
        self.pa_type = pa.uint32()

class UInt64Dtype(Dtype):
    def __init__(self):
        self.pa_type = pa.uint64()

class Int8Dtype(Dtype):
    def __init__(self):
        self.pa_type = pa.int8()

class Int16Dtype(Dtype):
    def __init__(self):
        self.pa_type = pa.int16()

class Int32Dtype(Dtype):
    def __init__(self):
        self.pa_type = pa.int32()

class Int64Dtype(Dtype):
    def __init__(self):
        self.pa_type = pa.int64()

class Float32Dtype(Dtype):
    def __init__(self):
        self.pa_type = pa.float32()

class Float64Dtype(Dtype):
    def __init__(self):
        self.pa_type = pa.float64()

class BooleanDtype(Dtype):
    def __init__(self):
        self.pa_type = pa.bool()

class Datetime64NSDtype(Dtype):
    def __init__(self):
        self.pa_type = pa.timestamp('ns')

class Datetime64USDtype(Dtype):
    def __init__(self):
        self.pa_type = pa.timestamp('us')

class Datetime64MSDtype(Dtype):
    def __init__(self):
        self.pa_type = pa.timestamp('ms')

class Datetime64SDtype(Dtype):
    def __init__(self):
        self.pa_type = pa.timestamp('s')

class StringDtype(Dtype):
    def __init__(self):
        self.pa_type = pa.string()

def make_dtype_from_string(obj):
    if obj in {'str', 'string', 'object'}:
        return StringDtype
    elif 'datetime' in obj:
        if obj == 'datetime64[ns]':
            return Datetime64NSDtype
        elif obj == 'datetime64[us]':
            return Datetime64USDtype
        elif obj == 'datetime64[ms]':
            return Datetime64MSDtype
        elif obj == 'datetime64[s]':
            return Datetime64SDtype
    elif 'int' in obj or 'Int' in obj:
        if obj in {'int', 'Int', 'int64', 'Int64'}:
            return Int64Dtype
        elif obj in {'int32', 'Int32'}:
            return Int32Dtype
        elif obj in {'int16', 'Int16'}:
            return Int16Dtype
        elif obj in {'int8', 'Int8'}:
            return Int8Dtype
        elif obj in {'uint64', 'UInt64'}:
            return UInt64Dtype
        elif obj in {'uint32', 'UInt32'}:
            return UInt32Dtype
        elif obj in {'uint16', 'UInt16'}:
            return UInt16Dtype
        elif obj in {'uint8', 'Uint8'}:
            return UInt8Dtype
    elif 'float' in obj:
        if obj in {'float64', 'Float64'}:
            return Float64Dtype
        elif obj in {'float32', 'Float32'}:
            return Float32Dtype
    elif 'bool' in obj:
        return BooleanDtype

def make_dtype_from_numpy(obj):
    np_to_pd_types = {v: k for k, v in pd_to_np_dtypes.items()}
    result = np_to_pd_types.get(obj)

def make_dtype_from_obj(obj):
    if isinstance(obj, np.dtype):
        return make_dtype_from_numpy(obj)
    elif isinstance(obj, str):
        return make_dtype_from_string(obj)

class CategoricalDtype(ExtensionDtype):
    def __init__(self, categories=None, ordered=None):
        """
        dtype similar to pd.CategoricalDtype with the categories
        stored on the GPU.
        """
        self._categories = self._init_categories(categories)
        self.ordered = ordered

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


class ListDtype(ExtensionDtype):
    def __init__(self, element_type):
        if isinstance(element_type, ListDtype):
            self._typ = pa.list_(element_type._typ)
        else:
            element_type = cudf.utils.dtypes.np_to_pa_dtype(
                np.dtype(element_type)
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
    def type(self):
        # TODO: we should change this to return something like a
        # ListDtypeType, once we figure out what that should look like
        return pa.array

    @property
    def name(self):
        return "list"

    @classmethod
    def from_arrow(cls, typ):
        obj = object.__new__(cls)
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
