import pandas as pd
import cudf
import numpy as np
from pandas.core.dtypes.dtypes import CategoricalDtype, CategoricalDtypeType

def is_bool_dtype(obj):
    # todo - pd.api.types.is_bool_dtype should not give false, nor work at all probably
    return isinstance(obj, cudf.BooleanDtype) or pd.api.types.is_bool_dtype(obj)

def is_datetime64_dtype(obj):
    return isinstance(obj, cudf.Datetime) or pd.api.types.is_datetime64_dtype(obj)

def is_timedelta64_dtype(obj):
    return isinstance(obj, cudf.Timedelta) or pd.api.types.is_timedelta64_dtype(obj)

def is_string_dtype(obj):
    return isinstance(obj, cudf.StringDtype) or (pd.api.types.is_string_dtype(obj) and not is_categorical_dtype(obj))

def is_numerical_dtype(obj):
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
    if isinstance(obj, cudf.Generic) and not isinstance(obj, cudf.CategoricalDtype):
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
