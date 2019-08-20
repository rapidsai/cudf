import numpy as np
import pandas as pd
from pandas.api.types import pandas_dtype
from pandas.core.dtypes.dtypes import CategoricalDtype, CategoricalDtypeType


def numeric_normalize_types(*args):
    """Cast all args to a common type using numpy promotion logic
    """
    dtype = np.result_type(*[a.dtype for a in args])
    return [a.astype(dtype) for a in args]


def is_categorical_dtype(obj):
    """Infer whether a given pandas, numpy, or cuDF Column, Series, or dtype
    is a pandas CategoricalDtype.
    """
    from cudf.core import Series, Index
    from cudf.core.column import Column, CategoricalColumn
    from cudf.core.index import CategoricalIndex

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
