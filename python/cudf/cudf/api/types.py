import numpy as np
import pandas as pd
from pandas.core.dtypes.dtypes import CategoricalDtype, CategoricalDtypeType

import cudf


def is_bool_dtype(obj):
    if hasattr(obj, 'dtype'):
        obj = obj.dtype
    # todo - pd.api.types.is_bool_dtype should not give false, nor work at all probably
    if hasattr(obj, "dtype"):
        obj = obj.dtype
    return isinstance(obj, cudf.BooleanDtype) or (not isinstance(obj, cudf.Generic) and pd.api.types.is_bool_dtype(
        obj
    ))


def is_datetime64_dtype(obj):
    if hasattr(obj, 'dtype'):
        obj = obj.dtype
    return isinstance(obj, cudf.Datetime) or (not isinstance(obj, cudf.Generic) and pd.api.types.is_datetime64_dtype(
        obj
    ))


def is_timedelta64_dtype(obj):
    if hasattr(obj, 'dtype'):
        obj = obj.dtype
    return isinstance(
        obj, cudf.Timedelta
    ) or (not isinstance(obj, cudf.Generic) and pd.api.types.is_timedelta64_dtype(obj))


def is_string_dtype(obj):
    if hasattr(obj, 'dtype'):
        obj = obj.dtype
    return isinstance(obj, cudf.StringDtype) or (not isinstance(obj, cudf.Generic) and (
        pd.api.types.is_string_dtype(obj) and not is_categorical_dtype(obj)
    ))


def is_integer_dtype(obj):
    if hasattr(obj, 'dtype'):
        obj = obj.dtype
    try:
        return isinstance(obj, cudf.Integer) or (not isinstance(obj, cudf.Generic) and pd.api.types.is_integer_dtype(obj))
    except:
        import pdb
        pdb.set_trace()


def is_numerical_dtype(obj):
    if hasattr(obj, 'dtype'):
        obj = obj.dtype
    if isinstance(obj, cudf.Generic):
        return isinstance(obj, (cudf.Number, cudf.BooleanDtype))
    if is_categorical_dtype(obj):
        return False
    if is_list_dtype(obj):
        return False
    return (
        np.issubdtype(obj, np.bool_)
        or np.issubdtype(obj, np.floating)
        or np.issubdtype(obj, np.signedinteger)
    )


def is_categorical_dtype(obj):
    """Infer whether a given pandas, numpy, or cuDF Column, Series, or dtype
    is a pandas CategoricalDtype.
    """
    if isinstance(obj, cudf.Generic) and not isinstance(
        obj, cudf.CategoricalDtype
    ):
        return False
    if obj is None:
        return False
    if isinstance(obj, cudf.CategoricalDtype):
        return True
    if obj is cudf.CategoricalDtype:
        return True
    if isinstance(obj, np.dtype):
        return False
    if isinstance(obj, CategoricalDtype):
        return True
    if obj is CategoricalDtype:
        return True
    if obj is CategoricalDtypeType:
        return True
    if isinstance(obj, str) and obj == "category":
        return True
    if isinstance(
        obj,
        (
            CategoricalDtype,
            cudf.core.index.CategoricalIndex,
            cudf.core.column.CategoricalColumn,
            pd.Categorical,
            pd.CategoricalIndex,
        ),
    ):
        return True
    if isinstance(obj, np.ndarray):
        return False
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
        if obj.type is CategoricalDtypeType:
            return True
    return pd.api.types.is_categorical_dtype(obj)


def is_list_dtype(obj):
    return (
        type(obj) is cudf.core.dtypes.ListDtype
        or obj is cudf.core.dtypes.ListDtype
        or type(obj) is cudf.core.column.ListColumn
        or obj is cudf.core.column.ListColumn
        or (isinstance(obj, str) and obj == cudf.core.dtypes.ListDtype.name)
        or (hasattr(obj, "dtype") and is_list_dtype(obj.dtype))
    )


def find_common_type(array_types=[], scalar_types=[]):
    array_types = [
        d.to_numpy if isinstance(d, cudf.Generic) else d for d in array_types
    ]
    scalar_types = [
        d.to_numpy if isinstance(d, cudf.Generic) else d for d in scalar_types
    ]

    return cudf.dtype(np.find_common_type(array_types, scalar_types))


def can_cast(from_, to, casting='safe'):
    if isinstance(from_, cudf.Generic):
        from_ = from_.to_numpy
    elif isinstance(from_, cudf.Scalar):
        from_ = from_.value
    if isinstance(to, cudf.Generic):
        to = to.to_numpy

    return np.can_cast(from_, to, casting=casting)


def result_type(*arrays_and_dtypes):

    arrays_and_dtypes = (
        d.to_numpy if isinstance(d, cudf.Generic) else d
        for d in arrays_and_dtypes
    )
    return cudf.dtype(np.result_type(*arrays_and_dtypes))

def isnan(obj):
    if isinstance(obj, cudf._lib.scalar.Scalar):
        obj = obj.value
    return np.isnan(obj)

def min_scalar_type(a):
    if isinstance(a, cudf.Scalar):
        a = a.value
    result = np.min_scalar_type(a)
    if result == np.dtype('float16'):
        return cudf.Float32Dtype()
    return cudf.dtype(result)

def promote_types(type1, type2):
    if isinstance(type1, cudf.Generic):
        type1 = type1.to_numpy
    if isinstance(type2, cudf.Generic):
        type2 = type2.to_numpy

    result = np.promote_types(type1, type2)
    if result == np.dtype('float16'):
        return cudf.Float32Dtype()
    return cudf.dtype(result)

def isscalar(element):
    return isinstance(element, cudf._lib.scalar.Scalar) or np.isscalar(element)
