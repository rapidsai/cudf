from cudf.utils.dtypes import cudf_dtypes_to_pandas_dtypes
import pandas as pd
import dask.dataframe as dd

def upcast_pandas_to_nullable(obj):
    if isinstance(obj, (pd.Series, dd.Series)):
        return obj.astype(cudf_dtypes_to_pandas_dtypes.get(obj.dtype, obj.dtype))
    elif isinstance(obj, (pd.DataFrame, dd.DataFrame)):
        for col in obj.columns:
            obj[col] = obj[col].astype(cudf_dtypes_to_pandas_dtypes.get(obj[col].dtype, obj[col].dtype))
        return obj
