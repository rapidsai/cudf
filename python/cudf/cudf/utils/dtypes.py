import numpy as np
import pandas as pd
from pandas.api.types import pandas_dtype
from pandas.core.dtypes.dtypes import CategoricalDtype, CategoricalDtypeType


def is_datetime_dtype(obj):
    if obj is None:
        return False
    if not hasattr(obj, "str"):
        return False
    return "M8" in obj.str


def is_categorical_dtype(obj):
    """Infer whether a given pandas, numpy, or cuDF Column, Series, or dtype
    is a pandas CategoricalDtype.
    """
    from cudf.dataframe import Series, Index
    from cudf.dataframe.column import Column
    from cudf.dataframe.index import CategoricalIndex
    from cudf.dataframe.categorical import CategoricalColumn

    if obj is None:
        return False
    if obj is CategoricalDtypeType:
        return True
    if isinstance(obj, str) and obj == "category":
        return True
    if hasattr(obj, "type") and obj.type is CategoricalDtypeType:
        return True
    if isinstance(
        obj,
        (
            CategoricalDtype,
            CategoricalIndex,
            CategoricalColumn,
            pd.Categorical,
            pd.CategoricalIndex,
        ),
    ):
        return True
    if isinstance(
        obj, (Index, Series, Column, pd.Index, pd.Series, np.ndarray)
    ):
        return is_categorical_dtype(obj.dtype)

    return pandas_dtype(obj).type is CategoricalDtypeType
